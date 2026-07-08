# log: FT 300-S 統合 (UR5e)

外付け Robotiq FT 300-S を UR5e で精度を確保しつつ, ロボットデータ (RTDE) と時間同期して取得する検討の記録. append-only.

## 2026-05-27 FT 300-S の接続方式・同期・遅延の確定

### 目的

UR5e 内蔵 F/T センサは精度不足のため, 外付け FT 300-S を使う. その力/トルクを, ロボットが RTDE で出す関節状態などと時間同期して取得したい (要求: 精度 + 同期).

### 確定した結論

- **e-Series コントローラは外付け FT 300-S を駆動しない**. Copilot URCap は手首内蔵センサ専用に設計されており, 外付けは公式非対応 (Robotiq DoF/Zendesk で確認). 実機でも USB をコントローラに挿すとセンサ LED が赤 (給電あり・通信なし) のまま. ライセンス (Copilot ドングル) の有無とは別問題で, ドングルがあっても外付けは駆動されない.
- 一方 **PC に直結すると LED が青 (通信確立)**. これが Robotiq 自身の推奨経路 (e-Series + 外付け + ROS は PC 直結で生データを扱え, との回答あり).
- 帰結: **ロボット側が外付けセンサ値を読めない以上, 単一クロック同期 (URScript → RTDE レジスタ注入) は物理的に不可**. PC 直読み + ソフト同期 (遅延補正) が唯一の現実解.

### PC 直読みの実装

- ドライバは既存 `robotiq_ft_sensor` (`underlay_ws/.../third_party/robotiq`) の `rq_sensor` ノード. ビルド済み.
- デバイス: FT 300-S = `/dev/ttyUSB1` (FTDI "USB TO RS-485", idProduct 6015). ttyUSB0 は別機器.
- トピック: `/robotiq_ft_wrench` (geometry_msgs/WrenchStamped, **N/Nm, 変換不要**, header.stamp = 受信時刻), 約 100 Hz.
- ゼロ点調整: `rosservice call /robotiq_ft_sensor_acc "command: 'SET ZRO'"` (ライセンス不要).
- **FTDI latency_timer**: 既定 16 ms だと実効レートが 62.5 Hz に律速される. 1 ms に下げると 100 Hz 回復. コンテナで udev が使えないため, `ft300s_sensor.launch` の launch-prefix でノード起動直前に sysfs へ `echo 1` して焼き込み (再起動後も自動).

### 遅延 (USB-PC vs RTDE)

- **FT (USB-PC) は robot (RTDE) に対し δ ≈ +15.5 ± 1.8 ms 遅れる** (Fz・onset で測定, latency_timer=1ms).
- 補正: `ft_msg.header.stamp -= rospy.Duration(0.015)`.
- 残留の床: 母集団 std ~4 ms. うち **2.9 ms は 100 Hz サンプリングの量子化** (= 10 ms 幅一様分布の std = 10/√12) で消せない. サブ 10 ms 同期はこの機材では不可 (センサ仕様の限界).
- 平均 δ は √N 平均 (タップ位相のランダム性がディザとして働く) で量子化より細かく出せる (標準誤差 = std/√N).

### FT 波形のリンギングの正体

- 打撃応答で FT は幅広くリンギングして見えるが, これはフィルタや工具のぶら下がりではなく, **~48 Hz の構造共振 (アーム/センサ) を 100 Hz でアンダーサンプリング/エイリアス**しているため (FFT で確認, robot 500 Hz では本来の高域が見える). FT の Nyquist は 50 Hz.

### 解析上の教訓 (FTA で確定)

- 軸振幅は `max-min` でなく **baseline (中央値) からのピーク偏差**で測る. `max-min` は対称リンギングする横軸を過大評価し, 片側に出る打撃軸 (Fz) を過小評価する (今回 "Fz が最小に見える" 違和感の原因だった. センサ・記録は健全).
- 遅延は `|f|` + cross-correlation でなく **単一軸 Fz の onset (立ち上がり)** で測る. `|f|` は整流でリンギングを歪め, ピーク法はリンギングの尾で +19〜25 ms と過大評価する.

### 成果物 (catkin_ws/src/osx_core/osx_ur5e/)

- `launch/ft300s_sensor.launch` — PC 直読み bring-up. latency_timer=1ms 焼き込み.
- `scripts/measure_ft_delay.py` — Fz・onset 方式の遅延測定 (δ, 標準誤差, 量子化床, per-tap プロット).
- `scripts/fft_ringing.py` — 打撃後リンギングの FFT (共振周波数同定, FT のエイリアス確認).
- `scripts/inspect_ft_taps.py` は measure_ft_delay.py に統合し削除.

### 次の作業

- δ ≈ 15 ms のタイムシフト再配信ノード (`/robotiq_ft_wrench` → `header.stamp -= 0.015` → `/robotiq_ft_wrench_synced`) + `message_filters` ApproximateTime (slop ~10 ms) でロボット状態と同期.

## 2026-06-08 FT 300-S キャリブレーション調査・実施

### 目的

UR5e + 外付け FT 300-S 構成で, 重力補償キャリブレーションの手順を確立する.

### 確定した事実

- **FT 300 / FT 300-S は e-Series 非対応** (CB-Series 専用). Robotiq Knowledge Base に明記:"Both are fully compatible with all CB-Series Model but are not compatible with E-Series Model." ([Difference Between the FT-300 and FT-300-S Sensors](https://blog.robotiq.com/knowledge/difference-between-the-ft-300-and-ft-300-s-sensors))
- FT 300-S サポートページの Copilot URCap は **CB-Series 向け**. e-Series 用の Copilot は内蔵センサ用ソフトウェア (別製品, USB ライセンスドングル必要).
- URCap のキャリブレーションウィザード (PolyScope Installation → FT Sensor) は e-Series + 外付け FT 300-S では使用不可.

### キャリブレーション方法 (URCap なし)

**方法 1: Visual Demo Software (PC スタンドアロン, 今回採用)**

- Windows PC + RS-485→USB 変換器 (ACC-ADT-RS485-USB) で FT 300-S に直結
- Visual Demo Software (SUI-1.2.5) の Tool Calibration タブで 3 姿勢キャリブレーション
  - X 軸下向き → Y 軸下向き → Z 軸下向き (各姿勢で外力なしの状態で記録)
- センサ内蔵加速度計を使い, ツール質量・重心・取付オフセットを自動算出
- **キャリブレーションデータはセンサ内部の不揮発メモリに永続保存** (電源 OFF でも保持). Robotiq 公式確認:"the calibration is actually saved within the sensor" (DoF),"Calibration data is stored permanently in the sensor and influences readings on all future startups" (Knowledge Base). FT 300-S でもソフトウェア・キャリブレーション方式は FT 300 と同一.

**方法 2: 自前 least squares 推定 (ROS)**

- `force_torque_tools` (KTH, ROS 1 Kinetic, メンテ停止) があるが, アルゴリズム自体は単純 (10 パラメータの線形最小二乗) なので必要なら自前実装可.
- 多姿勢 (10 以上推奨) でセンサ値 + 回転行列を記録し, バイアス・質量・CoG を推定.

### 永続化の検証結果

- 3 pose キャリブレーション実施後, 電源 OFF → ON, 同一姿勢でセンサ値を比較.
- 再起動後のオフセット差は約 1 N. キャリブレーションが消えていればツール重量分 (数 N〜数十 N) の差が出るはずなので, **永続化は確認済み**.
- 1 N の差はドリフトで説明可能 (仕様上, 5–10 分で 10–20 N ドリフトする. 出典: [Understanding Force Sensor Accuracy](https://blog.robotiq.com/knowledge/robotiq-ft-300-sensor-ft-300-s-sensor-and-copilot-software-general-measurements-5-1736280750218)).

### FT 300-S ドリフト仕様 (同上出典)

| 条件 | 典型的なオフセット |
|---|---|
| 時間経過 (5–10 分) | 10–20 N |
| 強い力印加後のヒステリシス | 約 3 N |
| 姿勢変化 (センサ自体由来) | 5 N 以上 |
| 信号ノイズ (Fz, 1σ, 1 秒) | 0.1 N |

- `SET ZRO` による一時ゼロ補正は揮発 (電源 OFF で消える). プログラム実行のたびに, できればサイクルごとに実行推奨.

### 再キャリブレーションが必要なタイミング

- センサをロボットから取り外して再取付したとき (ネジの締結応力が変わる)
- エンドエフェクタを交換したとき (質量・重心が変わる)

## 2026-06-29〜30: ft300s_measurement フレーム再定義・マニュアル値による控除

### ft300s_measurement フレーム再定義

旧定義: ft300s_sensor に対して xyz=(0,0,0.0065) rpy=(0,π,0). tool0 + 31mm (depression 面), 向き tool0 同一.
新定義: ft300s_sensor に対して xyz=(0,0,0) rpy=(π,0,π). tool0 + 37.5mm (tool contact surface), 向き tool0 同一.

変更理由: FT 300-S マニュアル p.35 Fig.5.1 の断面図で Z 軸原点が tool contact surface (fitting ring 上面) に描かれている. 6.5mm オフセットの公式裏付けが取れなかった.

### グリッパ慣性パラメータのマニュアル値追加

Robotiq 2F-85 マニュアル (General, 2020-02-11, p.82-83) から取得した値を `gripper.json` に `methods.manual` として追加.

- m = 0.921 kg, CoM = (0, 0, 60) mm from gripper mounting flange (= ft300s_measurement)
- TCP = (0, 0, 174) mm
- 慣性テンソル (指全開, kg·mm²): Ixx=4180, Iyy=5080, Izz=1250, 交差項=0
- 座標系: tool0 基準 = グリッパ締結フランジ基準

控除優先順位を manual > OLS+bias に変更 (`excitation_identifier.py`, `replay_excitation_trajectory.py`).

### FT 300-S マニュアルの座標系確認

FT 300-S Instruction Manual (TM Series, 2021-07-07):
- p.9:"Reference frame is centered on the Sensor. Z axis passes through the center of the depression with positive direction in the tool direction."
- p.35 Fig.5.1: Z 軸原点は tool contact surface (fitting ring 上面) に描画
- p.37: Thickness 41.5mm (overall), 37.5mm (from robot flange)
- p.38: CoM = (-6, 0, 13.5) mm, TCP = (0, 0, 37.5) mm, mass = 410g*, 慣性テンソル Ixx=316, Iyy=611, Izz=671 kg·mm²

CoM/TCP/慣性の座標系の z 原点は明示されていないが, TCP z=37.5mm = robot flange からの距離と一致することから, robot flange (tool0) 基準と推定.

### 合成データによる regressor フレーム基準の検証

Pinocchio の regressor が ft300s_measurement フレーム基準の hz を出力することを合成データで確認:
- Case A: regressor@ft300s_meas, wrench from phi_ft → hz 正確に復元 ✓
- Case C: regressor@tool0, wrench from phi_tool0 → hz 正確に復元 ✓
- Case B: regressor@ft300s_meas, wrench from phi_tool0 → hz は中間値 (不整合)

### 物体 CoM z の系統的過小推定 (未解決)

OLS+bias + マニュアル控除で物体 CoM z ≈ 121-125mm. 期待 ≈ 152mm (finger pad 位置). 差 ≈ 27mm.

状況:
- 質量は妥当 (obj m ≈ 0.36-0.37 kg, 期待 0.34 kg)
- Total hz ≈ 0.099 に対し期待 total hz ≈ 0.107. deficit = 0.008 kg·m
- グリッパ単体推定でも hz deficit ≈ 0.0074 (manual 0.05526 vs 推定 0.048)
- 物体の hz 寄与は正しく推定されており, deficit はグリッパ部分に起因

排除した仮説:
- 座標系不一致 (合成データで排除)
- FW 重力補償 (センサ単体でキャリブレーション済み, m_comp ≈ 0)
- wrench 軸対応 (rviz + 実物目視で確認済み)
- 把持位置 (アルミキューブで CoM 位置は明白)

未究明: 推定系がグリッパの hz を系統的に過小推定する原因. hz に主に寄与するのは FT の Tx, Ty 成分.

## 2026-06-22〜25 URDF 単一ソース化・フレームオフセット確定

### 目的

UR5e + FT 300-S + Robotiq 2F-85 の URDF を全パッケージで一貫して参照し, フレーム間オフセットを実測値に基づき確定する.

### 実施内容

#### URDF 単一ソース化

- `/robot_description` (ROS param server) を唯一の参照源とした. 各パッケージが別々の静的 URDF ファイルを参照していた構成を廃止.
- マスター xacro: `catkin_ws/src/osx_core/osx_ur5e/urdf/ur5e_ft300s_robotiq85.urdf.xacro`
- `resolve_robot_urdf_xml()` ユーティリティを iparam_identification と osx_3d_reconstruction に追加. `/robot_description` を優先し, 未設定の場合は xacro を展開してフォールバック.
- Pinocchio の読み込みを `buildModelFromUrdf(path)` から `buildModelFromXML(xml_string)` に変更 (全パッケージ).
- `[URDF-CHECK]` ログ出力を追加し, 3 回のフルテスト実行で単一ソースからの読み込みを確認.

#### フレーム名リネーム

- `robotiq_ft_frame_id` → `ft300s_measurement` に全パッケージでリネーム (iparam_identification, osx_3d_reconstruction, osx_mass_distribution).
- `ft300_sensor.launch` → `ft300s_sensor.launch` にリネーム.

#### URDF フレームオフセット (tool0-downward kinematic chain)

| リンク間 | xyz (m) | rpy (rad) | 備考 |
|---|---|---|---|
| FT300-S → tool0 | (0, 0, 0.0375) | (0, π, 0) | 0.0375 = カタログ高さ 0.0415 − フィッティング嵌入量 0.004 |
| ft300s_measurement → ft300s_sensor | (0, 0, 0.0065) | (0, π, 0) | 0.0065 はフィッティング嵌入量 |
| coupling → ft300s_sensor | (0, 0, 0) | (0, −π, 0) | オフセットなし |
| gripper → coupling_link | (0, 0, 0.011) | (0, −π/2, π/2) | |

`<origin xyz rpy>` の解釈: まず xyz で並進, 次に rpy (ZYX Euler) で回転.

### FT 300-S STL モデル寸法確認

- バウンディングボックス: X=89mm, Y=89mm, Z=42.2mm (xacro 内 scale=0.001 適用後).
- カタログ公称高さ 41.5mm に対し STL は 42.2mm (差 0.7mm はモデル末端処理の違いと推定).

### 成果物

- `catkin_ws/src/osx_core/osx_ur5e/urdf/ur5e_ft300s_robotiq85.urdf.xacro` — マスター xacro (フレームオフセット更新済み)
- `catkin_ws/src/osx_core/osx_ur5e/launch/ft300s_sensor.launch` — リネーム後の launch ファイル
