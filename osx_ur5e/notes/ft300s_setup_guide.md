# Robotiq FT 300-S セットアップガイド (UR5e)

UR5e (e-Series) で外付け Robotiq FT 300-S を使用するための手順書.
最終更新: 2026-06-08

## 1. 前提: e-Series との互換性

FT 300 / FT 300-S は **CB-Series 専用**であり, e-Series は公式非対応.

> "Both are fully compatible with all CB-Series Model but are not compatible with E-Series Model."
> — Robotiq Knowledge Base

具体的な影響:

- Copilot URCap のキャリブレーションウィザード (PolyScope Installation > FT Sensor) は使用不可
- FT 300-S サポートページに掲載されている URCap は CB-Series 向け
- e-Series 用の Copilot URCap は**内蔵センサ用**の別製品 (USB ライセンスドングル必要)

したがって, e-Series では **PC 直結 + ROS ドライバ**で運用する.

## 2. ハードウェア接続

### 必要なもの

| 部品 | 型番・仕様 |
|---|---|
| FT 300-S | Robotiq FTS-300-S-UR-KIT |
| RS-485 → USB 変換器 | Robotiq ACC-ADT-RS485-USB |
| 電源 | DC 5–24 V (USB 経由では給電されない) |

### 配線

1. FT 300-S の M12 コネクタからの線を RS-485 変換器に接続:
   - 白 (RS-485 A), 緑 (RS-485 B), シールド (GND) → 変換器
   - 赤 (24 V), 黒 (0 V) → 電源
2. 変換器の USB を **PC** に接続 (UR コントローラではない)
3. LED が**青**になれば通信確立 (赤のままなら給電のみで通信なし)

### デバイス確認

```bash
# FTDI "USB TO RS-485" (idProduct 6015) を探す
lsusb | grep FTDI
ls /dev/ttyUSB*
```

本環境では `/dev/ttyUSB1` に割り当てられている (ttyUSB0 は別機器).

## 3. キャリブレーション (重力補償)

### 3.1 Visual Demo Software による 3 姿勢キャリブレーション

URCap が使えない環境での公式キャリブレーション手段. **PC スタンドアロンで動作**し, PolyScope に依存しない.

#### 必要なもの

- Windows PC
- RS-485 → USB 変換器 (上記と同じもの)
- Visual Demo Software (SUI-1.2.5): [Robotiq サポートページ](https://robotiq.com/support/ft-300-force-torque-sensor)からダウンロード ([直接リンク](https://blog.robotiq.com/hubfs/support-files/SUI-1.2.5_20210303.zip))

#### 手順

1. FT 300-S を RS-485 → USB 変換器経由で Windows PC に接続
2. Visual Demo Software を起動
3. **Tool Calibration** タブを開く
4. 画面の指示に従い, ロボットを 3 姿勢に移動 (各姿勢で外力がかかっていないことを確認):

   | Step | 姿勢 |
   |---|---|
   | 1 | センサの **X 軸が下向き** |
   | 2 | センサの **Y 軸が下向き** |
   | 3 | センサの **Z 軸が下向き** |

5. ソフトウェアがセンサ内蔵加速度計を使い, ツール質量・重心・取付オフセットを自動算出
6. **Sensor Data** タブで補正後の読み値を確認

#### キャリブレーションデータの保存先

- **センサ内部の不揮発メモリに永続保存**される (電源 OFF でも保持)
- 以後のデータ出力は重力補償済みになる
- 電源再投入・ロボット再起動でも再キャリブレーションは不要 (実機で検証済み: 再起動後の差は約 1 N, ドリフトの範囲内)

#### 再キャリブレーションが必要なとき

- センサをロボットから**取り外して再取付**したとき (ネジの締結応力が変わる)
- **エンドエフェクタを交換**したとき (質量・重心が変わる)

### 3.2 自前 least squares 推定 (上級)

Visual Demo Software を使わず, ROS 上で多姿勢データから重力補償パラメータを推定する方法.

- 10 姿勢以上でセンサ値 (Fx, Fy, Fz, Tx, Ty, Tz) + エンドエフェクタの回転行列を記録
- 線形最小二乗法でバイアス (6 成分), ツール質量, 重心位置 (3 成分) を推定
- 推定パラメータでリアルタイムに重力成分を差し引く
- 既存パッケージ: `force_torque_tools` (KTH, ROS 1 Kinetic, メンテ停止). アルゴリズムが単純なため自前実装も容易

## 4. ROS ドライバの起動

### Launch ファイル

```bash
roslaunch osx_ur5e ft300s_sensor.launch
```

- ドライバ: `robotiq_ft_sensor` パッケージの `rq_sensor` ノード
- トピック: `/robotiq_ft_wrench` (geometry_msgs/WrenchStamped, 単位 N/Nm)
- レート: 100 Hz

### FTDI latency_timer の設定

FTDI の既定 latency_timer (16 ms) では実効レートが 62.5 Hz に律速される. `ft300s_sensor.launch` の launch-prefix で起動時に自動で 1 ms に設定:

```bash
# 手動で設定する場合
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
```

## 5. ゼロ補正 (SET ZRO)

### なぜ必要か

FT 300-S は温度変化等により継続的にドリフトする.

| 条件 | 典型的なオフセット |
|---|---|
| 時間経過 (5–10 分) | 10–20 N |
| 強い力印加後のヒステリシス | 約 3 N |
| 姿勢変化 (センサ自体由来) | 5 N 以上 |
| 信号ノイズ (Fz, 1 秒, 1σ) | 0.1 N |

### 実行方法

**ROS サービス経由:**

```bash
rosservice call /robotiq_ft_sensor_acc "command: 'SET ZRO'"
```

**URScript / ソケット経由:**

```python
socket_open("127.0.0.1", 63350, "acc")
socket_send_string("SET ZRO", "acc")
socket_close("acc")
```

### 運用ルール

- ゼロ補正は**一時的** (揮発). 電源 OFF で消える
- **プログラム実行の冒頭**, できれば**サイクルごと**に実行する
- 実行時, センサに外力が加わっていないことを確認する

## 6. RTDE との時間同期

FT 300-S (USB-PC) はロボット (RTDE) に対し **δ ≈ 15.5 ± 1.8 ms 遅れる**.

### 補正方法

```python
ft_msg.header.stamp -= rospy.Duration(0.015)
```

### 同期精度の限界

- 母集団の標準偏差: 約 4 ms
- うち 2.9 ms は 100 Hz サンプリングの量子化 (10 ms / √12) で原理的に消せない
- サブ 10 ms 同期はこの機材構成では不可

### 波形のリンギングについて

打撃応答で FT 波形がリンギングして見えるのは, ~48 Hz の構造共振 (アーム/センサ) を 100 Hz でエイリアスしているため. センサの異常ではない (FT の Nyquist は 50 Hz).

## 参考資料

### Robotiq 公式

- [FT 300-S 製品ページ](https://robotiq.com/products/ft-300-force-torque-sensor)
- [FT 300-S サポートページ (マニュアル・ソフトウェア)](https://robotiq.com/support/ft-300-force-torque-sensor)
- [Copilot ライセンス種別の解説](https://blog.robotiq.com/knowledge/copilot-licenses-distinctions)
- [FT 300 と FT 300-S の違い](https://blog.robotiq.com/knowledge/difference-between-the-ft-300-and-ft-300-s-sensors)
- [FT 300-S キャリブレーション操作ガイド](https://blog.robotiq.com/knowledge/operation-and-calibration-of-the-ft-300s-5-1736280819067)
- [FT 300-S 精度・ドリフト仕様](https://blog.robotiq.com/knowledge/robotiq-ft-300-sensor-ft-300-s-sensor-and-copilot-software-general-measurements-5-1736280750218)

### コミュニティ・その他

- [FT300 Sensor Calibration (DoF)](https://dof.robotiq.com/discussion/349/ft300-sensor-calibration)
