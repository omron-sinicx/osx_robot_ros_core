# Issues

## 双腕 EE 用 `osx_end_effector.urdf.xacro` の `gripper_tip` オフセット未測定

`osx_scene_description/urdf/components/osx_end_effector.urdf.xacro:4` で"The position of the TF frame has not been measured at all and needs to be calibrated."と明記された 0.163 m がそのまま. 単腕 (FT 300-S + 2F-85) は実測 211.4 mm で更新済みだが, 双腕側は別構成 (FT 無し + 0.01 m カプラ) のため別途実測が要る.

## D455 の launch 再起動での設定反映が未検証

d455_1/2/3 の RGB 設定 (exposure 31 / WB 4300 K / saturation 60) は dynamic_reconfigure で適用したもので、launch 経由の起動で同じ値になるかを確認していない。`rgb_lock_settings=true` の実装を信頼している状態。→ [2026-08-10_d455-rgb-tuning.md](LOGS/2026-08-10_d455-rgb-tuning.md)

## d455_0 (141322250927) の RGB 設定が未調整

未接続のため実測できず、未検証の旧値 (exposure 60 / WB 3700 / saturation 70) のまま。復帰時に `notes/debug/d455_rgb_color_calibration.md` の手順で再調整が必要。→ [2026-08-10_d455-rgb-tuning.md](LOGS/2026-08-10_d455-rgb-tuning.md)
