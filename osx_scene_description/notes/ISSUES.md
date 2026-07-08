# Issues

## 双腕 EE 用 `osx_end_effector.urdf.xacro` の `gripper_tip` オフセット未測定

`osx_scene_description/urdf/components/osx_end_effector.urdf.xacro:4` で"The position of the TF frame has not been measured at all and needs to be calibrated."と明記された 0.163 m がそのまま. 単腕 (FT 300-S + 2F-85) は実測 211.4 mm で更新済みだが, 双腕側は別構成 (FT 無し + 0.01 m カプラ) のため別途実測が要る.
