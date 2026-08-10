# D455 RGB 色調キャリブレーション手順

白い紙を基準に D455 の RGB 露出・ホワイトバランス・彩度を決める手順。
カメラを追加したとき、設置場所を変えたとき、照明条件が変わったときに再実行する。

初回実施: 2026-08-10 (d455_1/2/3)。結果は
`launch/osx_bringup_d455_<serial>.launch` に反映済み。

## 前提

- カメラが起動していること (`roslaunch osx_scene_description osx_bringup_d455s.launch`)
- 対象カメラの画角内に**白い紙** (コピー用紙で可) を置けること
- 調整中は紙とカメラを動かさないこと
- 照明を実運用と同じ状態にすること (窓のブラインド、室内灯の on/off を含む)

## 原理

RGB の設定は `dynamic_reconfigure` の `/<camera>/rgb_camera` サーバ経由で
ノード再起動なしに変更でき、その場でストリームへ反映される。

```bash
rosrun dynamic_reconfigure dynparam get /d455_1/rgb_camera
rosrun dynamic_reconfigure dynparam set /d455_1/rgb_camera exposure 31
```

ホワイトバランスの評価は、白い紙の領域で R/G と B/G を測り、
両方が 1.0 に近づくほど良いとする。誤差は `|R/G - 1| + |B/G - 1|`。

**重要**: 飽和 (クリップ) した領域は基準に使えない。8 bit の上限 255 に
貼り付いたチャンネルは真の値が分からないため、比が壊れる。
必ず露出を下げて飽和を解消してから色を測ること。

## 手順

### 0. 準備

オート露出とオートホワイトバランスを無効にする。これをしないと
設定が上書きされる。

```bash
CAM=d455_1
rosrun dynamic_reconfigure dynparam set /$CAM/rgb_camera enable_auto_exposure false
rosrun dynamic_reconfigure dynparam set /$CAM/rgb_camera enable_auto_white_balance false
```

### 1. フレームを取得して紙の位置 (ROI) を決める

```python
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
rospy.init_node('grab', anonymous=True, disable_signals=True)
m = rospy.wait_for_message('/d455_1/color/image_raw', Image, timeout=10)
img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, ::-1].copy()  # BGR
cv2.imwrite('frame.png', img)
```

保存した画像を見て、紙だけが入る矩形 `img[y0:y1, x0:x1]` を決める。
影や物体の縁を含めないこと。

### 2. 露出を下げて飽和を解消する

露出を粗く振り、紙の飽和率が 2% 以下になる範囲を見つける。

```python
clip = (roi.max(axis=-1) >= 254).mean() * 100   # 紙の飽和率 [%]
```

2026-08-10 の測定では、露出 34 で飽和率が 1.5% から 21% へ跳ね上がった。
この崖は急なので、崖の値の 1 割程度下を選ぶ。

### 3. ホワイトバランスを掃引する

飽和が消えた露出のまま、`white_balance` を 2800〜6500 K の範囲で
200 K 刻みに振り、誤差が最小の値を選ぶ。1 点あたり 1.2 秒の待機を入れる
(設定の反映に時間がかかるため)。

```python
b, g, r = [roi[:, :, i].mean() for i in range(3)]
err = abs(r/g - 1) + abs(b/g - 1)
```

### 4. 露出を詰める

WB を確定させたうえで露出を再度掃引し、下記の 3 つの釣り合いで決める。

| 指標 | 計算 | 望ましい方向 |
|---|---|---|
| 紙の輝度 | ROI のグレースケール平均 | 190〜210 |
| 紙の飽和率 | `(roi.max(-1) >= 254).mean()` | 2% 以下 |
| 暗部の潰れ | 画像全体で `(gray < 10).mean()` | 小さいほど良い |

露出を上げると暗部の潰れは減るが、白飛びが急激に増える。
崖の直前を選ぶ。

### 5. 彩度を決める

`saturation` は明るさを変えず、色差成分だけを増幅する。
色ノイズも一緒に増幅されるので、上げすぎない。

色ノイズは同一画素の複数フレーム間のばらつきで測る。

```python
S = np.stack([...])  # (N, H, W, 3) の float32
chroma = np.stack([S[:,:,:,2] - S[:,:,:,1], S[:,:,:,0] - S[:,:,:,1]], -1).std(axis=0).mean(-1)
dark = gray < 70
print(chroma[dark].mean())
```

2026-08-10 の測定 (暗部の色ノイズ、8 bit 階調):

| saturation | 色ノイズ | 備考 |
|---|---|---|
| 40 | 1.69 | |
| 55 | 2.05 | |
| **60** | **2.11** | 採用。55 からの増分はわずか 3% |
| 65 | 2.64 | ここから増分が急になる |
| 70 | 3.16 | D455 の既定 64 より高い。旧設定 |

60 と 65 の間に折れ点がある。60 までは彩度がほぼ無料で戻り、
65 からはノイズを払って買う区間になる。

### 6. 検証と反映

3 台以上ある場合、同じ照明下なら値を流用してよい。2026-08-10 の測定では
3 台の R/G・B/G が 1.0 の 5% 以内に収まり、個体差は無視できた。
ただし流用後も白い紙で必ず検証すること。

確定したら `launch/osx_bringup_d455_<serial>.launch` の
`rgb_exposure` / `rgb_white_balance` / `rgb_saturation` を書き換える。
`rgb_lock_settings=true` が AE/AWB を無効にしたうえでこれらを適用する。

## depth について

**depth は掃引しなくてよい。オート露出のままにする。**

- Stereo Module にホワイトバランスの設定は存在しない (IR は単色センサ)
- 露出の単位・範囲が RGB と異なる (RGB: 1〜10000 / depth: 1〜200000)。
  RGB の値をそのまま書き込んではいけない
- 2026-08-10 の掃引では、手動の最良値 (6000 µs, 有効画素 83.37%) が
  オート (83.35%) と同等で、手動化の利点がなかった
- 露出を上げると急激に悪化する。33000 µs では作業台の有効画素が 30.7% まで低下。
  環境光が積算されて IR ドットパターンが埋もれ、視差が取れなくなるため

## NeRF 用途での注意

- 視点間で露出・WB が変動すると、それ自体が大きな不一致になりフローターの原因になる。
  **AE/AWB の無効化が最も効果の大きい対策**
- 三脚固定でアームが物体を回す構成では、物体から見た照明方向が姿勢ごとに変わる。
  この輝度差は数十階調に達し、センサノイズ (0.2 階調程度) より桁違いに大きい。
  appearance embedding 等での対処が要る
- 静止させてから複数フレーム撮って平均すれば、ノイズは √N 分の 1 になる。
  カメラ固定なら視点の損失がないので、平均化に不利がない

## 参考

- 議事録: `notes/LOGS/2026-08-10_d455-rgb-tuning.md`
- 起動: `launch/osx_bringup_d455s.launch` (3 台を 5 秒刻みで起動)
- 個別設定: `launch/osx_bringup_d455_<serial>.launch` が値の真実
