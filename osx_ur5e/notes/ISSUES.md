# Issues

## `ur5e_controllers.yaml` の `end_effector_link` 不整合

`osx_ur5e/config/ur5e_controllers.yaml:137` の `cartesian_compliance_controller.end_effector_link` が `tool0` のまま. URDF / SRDF / Gazebo / 双腕コントローラはいずれも `gripper_tip_link` 基準 (前者 3 点との不整合). 単腕での力制御挙動が指先基準にならないため, フィンガー先端で接触力制御したい場合は `gripper_tip_link` に変更が必要.

## `ur5e_controllers.yaml` の payload 値が旧構成由来

`tool: mass: 1.380, cog: (0, −0.014, 0.064)` は git 履歴上 **FT 300-S 統合 (2026-05-27) より 7 ヶ月前 (2025-10-30)** の初版コミットで導入された値. 当時は FT 300-S 無し構成 (UR 内蔵 FT 利用) で, mass も 2F-85 単体 (0.925 kg) より 455 g 重く別ツール構成からの流用と推定. 現 FT 300-S + 2F-85 構成には流用不可で, `payload_estimation.launch` での再校正が必要.

## FT 300-S 慣性テンソルの公開値が無い

`robotiq_ft_sensor/urdf/robotiq_ft300.urdf.xacro` の mass / CoM は FT 300-S 仕様に更新済み (442 g, CoM (0, −6, −13.5) mm in sensor frame) だが, 慣性テンソル ixx/iyy/izz は FT 300 推定値を流用したまま (FT 300-S 公開値なし). Gazebo / 動力学 ID で慣性まで真面目に使うなら, CAD 推定か実測 (落下実験等) が必要.
