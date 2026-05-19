# ハンドアイキャリブレーション launch リファクタリング作業ログ

## セッション目的

`osx_moveit_config/launch/handeye_calibration.launch` を以下に沿って整理する:

1. パッケージ責務の分離: カメラ起動は `osx_scene_description` 側 (`osx_bringup_d455s.launch` 等) に任せ、本 launch は MoveIt + RViz HandEyeCalibration GUI のみを担当する
2. 単腕構成 (`ur5e_robotiq85.srdf` = `manipulator` + `gripper` 2 グループのみ) で MoveIt + RViz が確実に起動するようにする
3. 過去の dual-arm 構成由来の crash 要因を排除する

## 修正前の問題

`roslaunch osx_moveit_config handeye_calibration.launch camera_name:=d455_1` で以下が同時発生:

1. デフォルト引数が `camera_name:=d435i` で、d435i が未接続だと `The requested device with serial number 943222072344 is NOT found` が出続ける
2. RViz が `[FATAL] Group 'a_bot' was not found` で crash、`process has died (exit code -6)`
3. `[FATAL] Parameter '~moveit_controller_manager' not specified` の WARN/FATAL が連続

## デバッグで判明した crash 要因（3 段階）

### 要因 1: dual-arm 用 OMPL config の混入

`osx_moveit_config/launch/planning_pipeline.launch.xml` 経由でロードされる `osx_moveit_config/config/ompl_planning.yaml` が `a_bot:` / `b_bot:` / `ab_bot:` グループの planner 設定を含んでおり、単腕 SRDF と矛盾。

→ 対処: 単腕用に `manipulator:` と `gripper:` のみを定義した `ompl_planning_single_arm.yaml` を新規作成し、launch ファイル内でインライン load する。`planning_pipeline.launch.xml` の include は廃止。

### 要因 2: ROS master 上の dual-arm セッション残骸

過去の dual-arm セッションで loaded された `/move_group/planning_pipelines/ompl/a_bot/...` 等の params が param サーバに残存。`<rosparam command="load">` は既存 params を消さずに追加するだけのため、新しい single-arm yaml をロードしても古い a_bot 系 param が混在する。

→ 対処: 単腕 yaml ロード前に `<rosparam command="delete" param="/move_group/planning_pipelines/ompl"/>` で当該 namespace をクリーンアップ。

### 要因 3: `osx/osxSetupPanel` の SkillServer 経由 `MoveGroupInterface("a_bot")` 構築（決定打）

`osx_easy_handeye.rviz` の Panels セクションに含まれる `osx/osxSetupPanel` がロードされると、その内部メンバ `SkillServer ss_;` のコンストラクタ初期化リスト (`osx_skills/src/osx_skill_server.cpp:42-43`) で:

```cpp
SkillServer::SkillServer()
    : a_bot_group_("a_bot"), b_bot_group_("b_bot"),
```

を実行。SRDF に `a_bot` グループが存在しないため `libmoveit_move_group_interface.so` が
`Group 'a_bot' was not found.` をスローし、未捕捉例外で rviz プロセスが abort。

→ 対処: `osx_easy_handeye.rviz` をベースに `osxSetupPanel` と `moveit_task_constructor/Motion Planning Tasks` のパネル/ディスプレイ参照を除いた `handeye_calibration.rviz` を新規作成。launch から後者を参照する。`HandEyeCalibration` display 自体は保持（誤って削除した過程あり、最終的に line 1954-1995 を残す形で修正）。

## 最終的な変更ファイル

| ファイル | 状態 |
|---|---|
| `osx_moveit_config/launch/handeye_calibration.launch` | 修正: カメラ include 削除、`camera_serial_no` shim 削除、`<rosparam command="delete">` で ompl namespace クリア、単腕 OMPL config をインライン load、rviz config を新ファイルに切替 |
| `osx_moveit_config/config/ompl_planning_single_arm.yaml` | 新規: `manipulator` と `gripper` グループのみの OMPL planner 設定 |
| `osx_moveit_config/launch/handeye_calibration.rviz` | 新規: `osx_easy_handeye.rviz` から `osxSetupPanel` + Motion Planning Tasks パネル/display を除去した clean な rviz config |

## 使用方法

```bash
# 別ターミナル
roslaunch osx_ur5e connect_real_robot.launch
roslaunch osx_scene_description osx_bringup_d455s.launch

# キャリブレーション GUI
roslaunch osx_moveit_config handeye_calibration.launch camera_name:=d455_1
```

GUI 上で:
1. Displays リストに `HandEyeCalibration` が現れる（display プラグインであってパネルではない点に注意）
2. 選択すると右側 Property ペインに 3 タブが現れる: `Target` / `Context` / `Calibrate`
3. Context で `Sensor frame: d455_1_color_optical_frame`、`End-effector frame: tool0`、`Robot base frame: base_link`、`Planning group: manipulator` を設定
4. Target で ChArUco パラメータを実測値に合わせる
5. ロボットを 15〜20 姿勢に動かしながら **Calibrate タブの Take sample** を都度クリック (5 sample で自動計算開始、追加ごとに更新)
6. 結果を Save camera pose で書き出し

## 残る既知の制約

### `~moveit_controller_manager not specified` FATAL（運用に影響）

`move_group` の controller_manager プラグインが未設定のため、MoveIt の MotionPlanning panel から **Plan は可能だが Execute が機能しない**。`Plan & Execute` ボタンも Execute 部分が空振りする。

影響: 「rviz の MotionPlanning パネルでロボットを目標姿勢に動かしてから Take sample」のフローが不能。

回避策（現状の calibration 運用に十分）:
- UR Teach Pendant でジョグ → Take sample
- UR Free drive モード（重力補償）でロボットを手押し → Take sample
- 外部スクリプト（`osx_skills` 系）からロボット制御 → Take sample

恒久対応するなら以下を launch に追加する必要がある:
```xml
<param name="move_group/moveit_controller_manager"
       value="moveit_simple_controller_manager/MoveItSimpleControllerManager"/>
<rosparam command="load" file="$(find osx_moveit_config)/config/controllers_single_arm.yaml" ns="move_group"/>
```

ただし単腕用 `controllers_single_arm.yaml`（`scaled_pos_joint_traj_controller` を `manipulator` グループに紐づけ）の新規作成が前提。これは今セッションのスコープ外として残す。

## HandEyeCalibration プラグイン仕様の正確な理解（公式 doc 確認済み）

調査の過程で当初説明していた誤った理解を訂正:

- HandEyeCalibration display のタブは **`Target` / `Context` / `Calibrate` の 3 つのみ**
- 「Plan and Execute」タブや「Plan & Sample」自動巡回ボタンは **存在しない**
- sample 取得は **常に manual**: ロボットを別手段で動かしてから `Take sample` クリック
- 「Plan and Execute」は **MoveIt MotionPlanning rviz panel** の標準ボタンで、HandEyeCalibration とは別物（HandEye はそれを利用しない）
- 5 sample 集まると **自動で初回解算**、以降 sample 追加ごとに結果が更新される
- 精度は 12〜15 sample 程度でプラトーに到達

参照: [MoveIt Hand-Eye Calibration tutorial (PickNik Humble)](https://moveit.picknik.ai/humble/doc/examples/hand_eye_calibration/hand_eye_calibration_tutorial.html), [moveit_tutorials master](https://github.com/moveit/moveit_tutorials/blob/master/doc/hand_eye_calibration/hand_eye_calibration_tutorial.rst)

## 環境

- 日付: 2026-05-18
- librealsense: 2.54.2 (`ros-one-librealsense2`)
- D455 firmware: 5.17.0.10
- マザボ: MSI PRO B650-S WIFI (MS-7E26)
- CPU: AMD Ryzen 9 9950X3D
- Kernel: Linux 6.17.0-23-generic
