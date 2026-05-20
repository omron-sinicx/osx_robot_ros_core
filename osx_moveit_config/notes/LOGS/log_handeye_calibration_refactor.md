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

---

# 2026-05-19 セッション: 3 台キャリブ実施・D455 歪み補正・各種知見

## 実施結果: d455_1 / d455_2 / d455_3 を eye-to-hand キャリブし保存

3 台とも reprojection error 並進 ~8.5mm / 回転 ~0.8° で完了。結果を
`osx_scene_description/config/camera_calibration/base_link-to-d455_<N>_color_optical_frame.yaml`
に保存（identity placeholder を上書き）。カメラ再起動後、`publish_handeye_calibration.py`
が各 YAML を読み `base_link -> d455_<N>_color_optical_frame` の static TF として publish、
3 台とも保存値と一致を確認済み。

| カメラ | 並進 RMS | 回転 RMS | translation (x,y,z) |
|---|---|---|---|
| d455_1 | 8.0 mm | 0.87° | (-0.4143, 0.6634, 0.5367) |
| d455_2 | 8.7 mm | 0.78° | ( 0.3696, 0.9835, 0.3863) |
| d455_3 | 8.5 mm | 0.76° | ( 0.4745, 0.4399, 0.3136) |

運用知見:
- カメラを切り替えるたびに Calibrate タブの **Clear samples** が必須（前カメラのサンプル混入で AX=XB が発散）。Clear はソース上も全バッファ(effector_wrt_world_/object_wrt_sensor_/joint_states_/tree_view_model_)を消去するため確実。
- 当初 d455_2 が 14.2mm と高かったが、**回転多様性(roll/pitch/yaw を各 ±30°以上)を意識して再サンプル**したら 8.7mm に改善。サンプル数より姿勢多様性が効く。
- HandEye plugin の solve 結果は rviz 稼働中のみ `/tf` に動的 publish される。**rviz を閉じると static publisher の旧値に戻る**ため、必ず solve 後に YAML へ保存すること。

## reprojection error の単位ラベルが GUI で逆転している（バグ）

`handeye_control_widget.cpp:412` が `getReprojectionError` の戻り値
（`handeye_solver_base.h:141` は `(rotation[rad], translation[m])` 順）に対して
`first` に "m"、`second` に "rad" を付けており、ラベルが逆。
表示 `X m, Y rad` の実体は X=回転[rad]、Y=並進[m]。

GitHub issues に該当報告は見つからず（#139 等は別件）。詳細・修正案は
プロジェクトルートの `moveit_calibration_reprojection_error_label_bug.md` に記録。

## ChArUco "longest board side (m)" の定義

`handeye_target_charuco.cpp:191` の `square_size = board_size_meters / max(squares_x, squares_y)`
より、**マージン（白縁）を含まないチェッカーマトリクス本体の最長辺寸法**。
margin は印刷画像生成 (`createTargetImage`) でのみ使われ、姿勢推定には不使用。

## D455 color 歪み補正ノードを追加

D455 color sensor は `RS2_DISTORTION_INVERSE_BROWN_CONRADY` だが realsense2_camera は
camera_info を `plumb_bob` ラベルで素通し publish するため、OpenCV/cv::aruco 系が
逆方向に歪み補正し、画角端でマーカー検出が歪む。対策として
`osx_scene_description/scripts/rectify_d455_color`（pyrealsense2 の
rs2_deproject/project を使い正しい Inverse Brown-Conrady で undistortion map を構築 →
cv2.remap）を新規作成。`image_rect` + D=0 の `camera_info_rect` を publish し、
HandEye の image/camera_info topic をこれに向ける。pyrealsense2 は pip で導入。

## D455 60fps と手動露出のトレードオフ

`camera_fps:=60` でも color は 30fps 止まり。原因は **手動露出
(`enable_auto_exposure=false`)** で、Intel 公式も「auto-exposure を切ると fps が無効化」
と記載。`enable_auto_exposure=true` にすると 60fps 復帰（実機確認済み）。depth は影響なく
60fps 出る。色味ロック(複数カメラの色一致)と 60fps は D455 では両立不可。

なお `osx_bringup_d455s.launch` には `rgb_lock_settings` 引数が無く、個別 launch が
`value="true"` ハードコードのため、**CLI から `rgb_lock_settings:=false` は効かない**
（色味ロック=30fps 固定）。60fps 化には個別 launch 編集が必要。

## handeye_calibration.launch から dead な camera_name 引数を削除

リファクタでカメラ起動を分離した結果 `camera_name` は launch 内で未参照になっていた
（カメラ選択は rviz Context タブで行う）。混乱の元になるため削除。起動は
`roslaunch osx_moveit_config handeye_calibration.launch`（引数なし）でよい。

## 環境（2026-05-19）

- librealsense 2.54.2 / pyrealsense2 2.57.7（pip）/ D455 FW 5.17.0.10
- MSI PRO B650-S WIFI (MS-7E26) / AMD Ryzen 9 9950X3D / Linux 6.17.0-23-generic
