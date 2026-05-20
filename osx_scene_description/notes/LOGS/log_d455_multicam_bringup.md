# D455 マルチカメラ同時起動デバッグ作業ログ

## 環境

| 項目 | 値 |
|---|---|
| マザボ | MSI PRO B650-S WIFI (MS-7E26) |
| BIOS | 1.L0 (2025-06-19) |
| CPU | AMD Ryzen 9 9950X3D |
| Kernel | Linux 6.17.0-23-generic |
| librealsense | 2.54.2 (ros-one-librealsense2 deb) |
| realsense2_camera | ROS one wrapper |
| D455 firmware | 5.17.0.10（librealsense 推奨 FW: 5.15.1） |
| カメラ台数 | D455 × 3（最大 4 台運用想定） |

## カメラとシリアル番号のマッピング

| 名前 | シリアル | 個別 launch |
|---|---|---|
| d455_0 | `141322250927` | `osx_bringup_d455_141322250927.launch` |
| d455_1 | `142422250629` | `osx_bringup_d455_142422250629.launch` |
| d455_2 | `146222253334` | `osx_bringup_d455_146222253334.launch` |
| d455_3 | `138322252534` | `osx_bringup_d455_138322252534.launch` |

## 観測された問題

3 台以上の D455 を `timed_roslaunch` でステージング起動した際、必ず 1 台がフレーム配送に失敗する症状:

- `realsense2_camera` nodelet は alive
- 該当カメラの image topic は registered だが publish ゼロ
- librealsense backend ログに `backend-v4l2.cpp:1510 Frames didn't arrived within 5 seconds` が 5 秒毎に出力
- 起動時の typical errors:
  - `Left MIPI error` (D4 ASIC 内部 MIPI バスエラー)
  - `hwmon command 0x75( 0 1df 0 27f ) failed (response -6= Invalid parameter)`
  - `get_xu(ctrl=1) failed! Last Error: Device or resource busy`

## デバッグ過程（FTA を適用）

### Phase 1: インターバル sweep（N=10〜60s, 5s 刻み）

ハブ経由構成（UGREEN 40522 経由 → 後にマザボ内蔵ハブ経由）で実施。すべての N 値で 1〜2 台が失敗。失敗カメラは d455_1 と d455_3 で trial 毎に交互、d455_2 は常成功。

| N (s) | d455_1 | d455_2 | d455_3 |
|---|---|---|---|
| 10 | ✗ | ✓ | ✗ |
| 15 | ✓ | ✓ | ✗ |
| 20 | ✗ | ✓ | ✓ |
| 25 | ✓ | ✓ | ✗ |
| ... | (以下同じパターン) | | |

→ **インターバル拡大では解消不可** と判明。

### Phase 2: `initial_reset:=true` 全カメラ付き

全カメラに hw_reset を強制。4/4 trial で d455_1（最初に起動するカメラ）が連続失敗。

→ **A4 (firmware state)** 仮説除外。`initial_reset` の hw_reset cascade は他カメラの streaming を巻き込み逆効果。

### Phase 3: 環境バージョン確認

- librealsense 2.54.2 と FW 5.17.0.10 のミスマッチ（mismatch 警告は出るが streaming 自体には致命的でない）
- 慢性的 warning なので間欠失敗の主因とは説明力不足

### Phase 4: 物理 USB トポロジ調査（決定打）

`lsusb -t` で実態を確認。**「直挿し」のつもりが 2/3 のカメラがマザボ内蔵 USB ハブ経由**だった:

```
PCI 0000:0e:00.0 (B650 chipset ASMedia ASM3242, USB 3.2 Gen 2x2)
 └─ Bus 02 Port 5: ASM107x 4-port hub @ 5 Gbps   ← マザボ内蔵 hub
     ├─ Port 1: D455 (d455_3)
     └─ Port 2: D455 (d455_1)

PCI 0000:10:00.4 (AMD chipset xHCI, USB 3.1 Gen 1)
 └─ Bus 06 Port 1: D455 (d455_2)   ← 直結（hub 非経由）
```

MSI PRO B650-S WIFI の背面 USB-A 5Gbps ポート群（マニュアル page 24「#5」、4 個）はすべて ASM1074 ハブ配下。背面 USB-A 10Gbps（#4, #9, #10）は CPU 直結。

## Root cause（確定）

**マザボ内蔵 ASMedia ASM1074 USB ハブ上で 2 台以上の D455 が同時に startup negotiation する時の USB control transfer (EP0) の serialization 競合**。

### 機序

1. 各 D455 起動時に大量の EP0 control transfer（UVC controls、XU、hwmon コマンド）が発生
2. 共有ハブ上の同時 transfer は内部 arbitration で serialize される
3. 1 台のコマンドが timeout / EBUSY 応答 → librealsense backend が stream pipeline 確立に失敗
4. 該当カメラだけが `Frames didn't arrive` ループに陥る

### 非対称性の説明

- d455_2 は単独 controller 配下 → 競合なし、常成功
- d455_1 / d455_3 は ASM1074 配下で共有 → 1 台が必ず失敗
- 失敗対象がランダム = レース勝者がタイミング次第

## 解決策

**全 D455 をハブ非経由の USB ポートに分散**する（ハブ配下のクライアントを 0 もしくは 1 台に保つ）。

### MSI PRO B650-S WIFI 背面 USB ポートマップ（マニュアル page 24）

| ポート | Type | Speed | Source | D455 利用可否 |
|---|---|---|---|---|
| #4 | USB-A | 10 Gbps | CPU 直結 | ✓ 推奨 |
| #5 グループ (4 個) | USB-A | 5 Gbps | ASM1074 hub 配下 | ✗ 使うな（複数台時）<br>△ 1 台のみ単独で可 |
| #9 | USB-A | 10 Gbps | CPU 直結 | ✓ 推奨 |
| #10 | USB-A | 10 Gbps | CPU 直結 | ✓ 推奨 |
| #11 | USB-C | 20 Gbps | B650 chipset 直結 | ✓ 推奨（USB-C↔USB-C ケーブルが必要） |

### 4 台運用時の推奨配置

| カメラ | ポート | 経路 |
|---|---|---|
| d455_1 | #4 | CPU 直結 (PCI 10:00.3) |
| d455_2 | #9 | CPU 直結 (PCI 10:00.4) |
| d455_3 | #10 | CPU 直結 (別 PCI) |
| d455_4 | #11 (USB-C) | B650 chipset 直結 (PCI 0e:00.0) |

USB-C 利用には USB-C↔USB-C 3.x ケーブル（5 Gbps 以上対応品）が必要。D455 同梱の USB-C↔USB-A 3.0 ケーブルでは #11 を 5 Gbps として使うことに。

### 各ポートの電力供給

- USB-A 3.x (#4/#9/#10/#5): 5V / 900 mA（USB-IF 仕様）→ D455 の 720 mA 申告に対し 80% 利用、余裕 180 mA
- USB-C (#11): 5V / 3000 mA（USB-C デフォルト）→ 同 24% 利用、余裕 2280 mA

D455 の実測ピーク消費 ~570 mA（datasheet Table 7-7, 7-8）。3〜4 台同時稼働でも各ポート定格内に収まる。

## 検証結果

| 構成 | trial | PASS | 結論 |
|---|---|---|---|
| ハブ共有, N=10〜60s, no_reset | 11 | 0/11 | インターバル無効 |
| ハブ共有, N=15s, initial_reset 全台 | 4 | 0/4 | hw_reset は逆効果 |
| **ハブ非経由, N=15s** | 5 | **5/5** | hub 共有が真因と確定 |
| **ハブ非経由, N=1s** | 5 | **5/5** | インターバルは事実上不要 |

## 運用ルール

1. **D455 を接続する前に `lsusb -t` で配置を確認する**
   - 各カメラが root hub の直下（`Bus NN root` の直下）にあること
   - `hub/4p` の配下にいないこと、または配下にいる場合は単独であること
2. **同じハブ配下に複数の D455 を置かない**
3. **`osx_bringup_d455s.launch` の timed_roslaunch 値は 5 / 10 / 15 を維持**
   - ハブ非経由構成では N=1s でも動作するが、物理構成変更時の保険として残す
4. **eye-to-hand ハンドアイキャリブレーション**は `osx_moveit_config/launch/handeye_calibration.launch` を使い、`camera_name:=d455_<N>` で指定（d455_0..3 のシリアル分岐を内蔵）

## 副次 finding（運用上の留意点）

### librealsense 2.54.2 + D455 FW 5.17.0.10 の version mismatch warnings

以下の警告は出るが streaming 自体には致命的でなく、無視可能:

- `hwmon command 0x75( 0 1df 0 27f ) failed (response -6= Invalid parameter)`
- `UVC non compliance: permanently disabling control 981ae2 (Region of Interest Auto Ctrls), due to error -5`

影響: ROI Auto 機能が無効化される。固定 ROI / 手動 exposure 運用では問題なし。

### `Left MIPI error` warning

D455 内部 D4 ASIC の transient 報告で、stream は続行可能。多発する場合は USB 帯域 / 電力の見直しを検討。

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `osx_scene_description/launch/osx_bringup_d455s.launch` | 4 台一括起動オーケストレータ（`enable_d455_N` フラグ・timed_roslaunch） |
| `osx_scene_description/launch/osx_bringup_d455_<serial>.launch` × 4 | 個別カメラの bringup（シリアル → camera_name マッピング、RGB ロック値） |
| `osx_scene_description/launch/osx_bringup_single_camera.launch` | 汎用 single-camera launch（`calibration_parent`/`calibration_child` 引数、`rgb_lock_settings` ブロック） |
| `osx_scene_description/launch/osx_bringup_cameras.launch` | dual-camera 用 helper（D455 直接運用では未使用、互換用） |
| `osx_scene_description/config/camera_calibration/base_link-to-d455_<N>_color_optical_frame.yaml` × 4 | eye-to-hand キャリブ結果 |
| `osx_moveit_config/launch/handeye_calibration.launch` | eye-to-hand キャリブ実行 GUI（d455_0..3 シリアル分岐内蔵） |

---

## 追記 (2026-05-19): D455 color 歪み補正ノード

`scripts/rectify_d455_color` を追加（D455 color の Inverse Brown-Conrady 歪みを
pyrealsense2 で正しく補正し `image_rect` を publish）。背景・使い方の詳細は
`osx_moveit_config/notes/LOGS/log_handeye_calibration_refactor.md`（2026-05-19 セッション）参照。
