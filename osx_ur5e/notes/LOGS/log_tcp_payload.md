# log: TCP / Payload (UR5e + FT 300-S + Robotiq 2F-85)

UR5e + FT 300-S + Robotiq 2F-85 構成における TCP 定義・ペイロード (mass / CoM) の整理と URDF / コントローラ / ペンダント間の整合確認の記録. append-only.

## 2026-05-29 TCP / Payload の整理と URDF 質量パラメータ更新

### TCP の出どころと運用整理

- 本プロジェクトでは **`gripper_tip_link` を TCP として一貫使用**. URDF・MoveIt SRDF・Gazebo・双腕コントローラいずれも `gripper_tip_link` 基準 (例: `osx_ur5e/urdf/ur5e_ft300s_robotiq85.urdf.xacro:63-67`, `osx_ur5e/config/ur5e_robotiq85.srdf:5`).
- 例外: **`osx_ur5e/config/ur5e_controllers.yaml:137` の `cartesian_compliance_controller.end_effector_link` だけが `tool0` のまま**. Gazebo (`1_bot_controllers.yaml`) や双腕 (`a_bot_controllers.yaml`) は `gripper_tip_link` に揃っているので, 単腕力制御の挙動も `gripper_tip_link` に合わせるべき (修正候補).
- UR の **`tool0` 原点はフランジのフラット面 (M6 タップ穴が並ぶ環状の面) の中心**. 中央の φ31.5 位置決めボス (約 6 mm 突出) は基準ではない. 実測は M6 タップ穴周辺の環状面にプローブを当てる. UR DH パラメータの `d6 = 99.6 mm` がこのフラット面までの距離.

### tool0 → gripper_tip の実測 + URDF 更新

- 実機 (FT 300-S + 2F-85) で tool0 → 指先を測定: **211.4 mm**.
- URDF (`osx_ur5e/urdf/ur5e_ft300s_robotiq85.urdf.xacro`) の `gripper_tip_joint` オフセットを **0.163 → 0.1699 m** に更新. 内訳: 41.5 mm (FT 300-S 厚) + 169.9 mm (`robotiq_85_base_link` → tip).
- 元の 0.163 m は xacro コメントで「未測定, 要キャリブ」と明記された heritage 値 (o2ac-ur 由来). 実物のフィンガーパッド先端 (Robotiq 2F-85 仕様 148–162 mm) と比較しても今回の実測値の方が妥当.
- **双腕用 `osx_scene_description/urdf/components/osx_end_effector.urdf.xacro:47` は 0.163 のまま**. FT 300-S 無し構成のため別途実測が要る (本セッションでは未対応).

### URDF 質量パラメータの整合化

`ur_robot_driver` / `robot_state_publisher` は EE 慣性を見ないが, Gazebo シミュ / 動力学 ID 用途で値の信頼性が要るため, 第三者パッケージの inertial を実機相当値に書き換え:

- **FT 300-S** (`underlay_ws/.../robotiq_ft_sensor/urdf/robotiq_ft300.urdf.xacro`):
    - 元: mass = 0.300 kg (FT300 値), CoM = (0, 0, −17) mm (sensor frame).
    - 更新: **mass = 0.442 kg**, **CoM = (0, −6, −13.5) mm** (sensor frame). マニュアル §5.2.1 の実重 + 計測フレームでの CoM (−6, 0, 13.5) mm を `ft300_sensor` フレームへ rpy=(0, π, −π/2) で座標変換.
    - 慣性テンソルは FT 300-S 公開値が無いため FT 300 推定を流用 (コメント記載).
- **Robotiq 2F-85** (`underlay_ws/.../robotiq-cri/robotiq_description/urdf/robotiq_arg2f.xacro`):
    - 元: `robotiq_arg2f_base_link` mass = 0.22652 kg (CAD 由来). 指リンク 4×2 = 0.137 kg, パッド 0 kg, 合計 0.364 kg (カタログ 0.925 kg に対し ~560 g 不足).
    - 更新: **base mass = 0.788 kg** (= 0.925 − 0.137). モータ・ギア・電子部品分の不足を base に集約.
    - 指リンクと CoM/慣性は CAD 値のまま. ベース集中はやや乱暴だが, 力制御の重力補償は合計質量と CoM 位置だけ効くため実害は小さい.
- 親 xacro `ur5e_ft300s_robotiq85.urdf.xacro` のヘッダコメントを「inertials are FT300, left as-is」から「override 済み (FT300-S 値 + 2F-85 カタログ整合)」に更新.

### ペンダントの TCP / Payload 設定値

- **ペンダント (Installation) と ROS (URDF / controllers) は独立**. 両方に同等値を入れて初めて整合する.
- 単位は Imperial 設定下では lb / inch.
- **TCP**: X=0, Y=0, **Z=8.323"** (=211.4 mm), Rx=Ry=Rz=0°.
- **Payload (方針)**: プロジェクト推奨は **`force_torque_tools` (`payload_estimation.launch`) で実測キャリブ → `*_ft_calib_data.yaml`**. ペンダント Payload は 0 にしておくことで二重補償を防ぐ (`osx_scene_description/README.md:7-15` 既存方針).
- もしペンダント側で補償したい場合の参考値: mass ≈ **3.04 lb** (= 1.367 kg = 0.442 + 0.925), CoM ≈ (0, −0.08", 2.80") (= (0, −2, 71) mm, URDF 由来計算値).

### `ur5e_controllers.yaml` の既存 payload 値の精査

- `tool: mass: 1.380, cog: (0, −0.014, 0.064)` (= (0, −14, 64) mm).
- git 履歴: **初版コミット `5220f04` (2025-10-30)** で導入. 一方 FT 300-S 統合は **`37634a3` (2026-05-27)** が初出. つまり **この値が入った時点で FT 300-S は物理構成に無かった**.
- 初版 yaml は `ft_sensor_ref_link: "wrist_3_link"` (UR 内蔵 FT 参照) でコメントは "mass of everything that's mounted after the sensor". 当時の "after the sensor" は tool0 から先全体を指していたはず.
- ただし 1.380 kg は 2F-85 単体 (0.925 kg) より 455 g 重く, 2F-85 単独の校正値ではない. o2ac-ur 等の流用 or 別ツール (スクリュードライバ等) が付いていた構成からの継承の可能性.
- **結論: 現在の FT 300-S + 2F-85 構成には流用不可**. 再校正が必要.

### 次の作業

- `ur5e_controllers.yaml:137` の `end_effector_link` を `tool0` → `gripper_tip_link` に修正 (他コンフィグと整合).
- `payload_estimation.launch` を新構成 (FT 300-S + 2F-85, tip=211.4 mm) で実行 → `tool: mass / cog` を実測値に更新.
- 双腕 `osx_end_effector.urdf.xacro:47` の `gripper_tip` オフセット 0.163 m を実測 → 更新.
- ペンダント `Installation → TCP` に Z=8.323" を投入 (Default + Active 両方).
