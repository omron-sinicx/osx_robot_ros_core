# TODO

## FT 300-S 統合 (UR5e)

- [x] FT 300-S の接続方式を確定 (e-Series コントローラ非対応 → PC 直読み)
- [x] PC 直読み bring-up (`ft300_sensor.launch`, latency_timer=1ms で 100 Hz)
- [x] USB-PC vs RTDE の遅延測定 (δ ≈ 15 ms, `measure_ft_delay.py`)
- [x] FT リンギングの原因究明 (~48 Hz 構造共振の 100 Hz エイリアス, `fft_ringing.py`)
- [ ] δ=15 ms タイムシフト再配信ノード + ApproximateTime 同期の実装
- [ ] (任意) 同期後の残留誤差をアプリ要件 (許容同期精度) に照らして検証
- [x] 3 pose キャリブレーション (Visual Demo Software) 実施・永続化確認済み
- [ ] (任意) ゼロ化 (`SET ZRO`) の運用手順整理 (プログラム冒頭・サイクルごとの自動ゼロ補正)

## URDF 単一ソース化 (UR5e + FT 300-S + Robotiq 2F-85)

- [x] `/robot_description` (ROS param server) を唯一の URDF 参照源に統一 (iparam_identification, osx_3d_reconstruction, osx_mass_distribution)
- [x] `robotiq_ft_frame_id` → `ft300s_measurement` に全パッケージでリネーム
- [x] `ft300_sensor.launch` → `ft300s_sensor.launch` にリネーム
- [x] `resolve_robot_urdf_xml()` ユーティリティ追加 (iparam_identification, osx_3d_reconstruction): `/robot_description` を優先, フォールバックは xacro 展開
- [x] Pinocchio を `buildModelFromUrdf(path)` → `buildModelFromXML(xml_string)` に変更 (全パッケージ)
- [x] URDF フレームオフセット設定 (tool0-downward kinematic chain):
  - FT300-S → tool0: xyz=(0,0,0.0375) rpy=(0,π,0)
  - ft300s_measurement → ft300s_sensor: xyz=(0,0,0) rpy=(π,0,π)
  - coupling → ft300s_sensor: xyz=(0,0,0) rpy=(0,−π,0)
  - gripper → coupling_link: xyz=(0,0,0.011) rpy=(0,−π/2,π/2)
- [x] `[URDF-CHECK]` ログによる単一ソース確認 (3 回フルテスト実行)

## TCP / Payload (UR5e + FT 300-S + 2F-85)

- [x] tool0 → gripper tip の実測 (211.4 mm) と URDF (`ur5e_ft300s_robotiq85.urdf.xacro`) の `gripper_tip_joint` 更新 (0.163 → 0.1699 m)
- [x] FT 300-S URDF inertial の FT300 値 → FT300-S 実測値への置換 (mass 0.442 kg, CoM (0, −6, −13.5) mm in sensor frame)
- [x] Robotiq 2F-85 ベース mass のカタログ整合化 (0.22652 → 0.788 kg, 合計 0.925 kg)
- [ ] `ur5e_controllers.yaml:137` の `end_effector_link` を `tool0` → `gripper_tip_link` に統一
- [ ] `payload_estimation.launch` を新構成 (FT 300-S + 2F-85) で実行し `tool: mass / cog` を再校正
- [ ] 双腕 `osx_end_effector.urdf.xacro:47` の `gripper_tip` オフセット (0.163 m, 未測定) を実測 → 更新
- [ ] ペンダント `Installation → TCP` を Z=8.323" (=211.4 mm), Rx=Ry=Rz=0 で設定 (Default + Active)
