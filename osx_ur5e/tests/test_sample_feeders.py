"""Unit tests for BagCursor and the pure decode helpers (no ROS runtime)."""

from types import SimpleNamespace

import numpy as np
import pytest

from osx_ur5e.sample_feeders import (
    JOINT_ORDER,
    BagCursor,
    decode_image_msg,
    joint_state_to_value,
    reorder_joint_state,
    stamp_of,
)


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def test_reorder_joint_state_from_alphabetical():
    # UR driver publishes alphabetically; values here encode the target index.
    alphabetical = [
        "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    values = [2.0, 1.0, 0.0, 3.0, 4.0, 5.0]
    out = reorder_joint_state(alphabetical, values)
    np.testing.assert_allclose(out, np.arange(6, dtype=float))


def test_joint_state_to_value_concatenates_qpos_qvel():
    msg = SimpleNamespace(
        name=list(JOINT_ORDER),
        position=list(range(6)),
        velocity=[v * 0.1 for v in range(6)],
    )
    out = joint_state_to_value(msg)
    assert out.shape == (12,)
    np.testing.assert_allclose(out[:6], np.arange(6, dtype=float))
    np.testing.assert_allclose(out[6:], np.arange(6, dtype=float) * 0.1)


def test_joint_state_to_value_no_velocity():
    msg = SimpleNamespace(name=list(JOINT_ORDER), position=list(range(6)), velocity=[])
    out = joint_state_to_value(msg)
    np.testing.assert_allclose(out[6:], np.zeros(6))


def test_decode_raw_image_rgb8_passthrough():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    msg = SimpleNamespace(encoding="rgb8", height=2, width=3, data=img.tobytes())
    np.testing.assert_array_equal(decode_image_msg(msg), img)


def test_decode_raw_image_bgr8_canonicalized_to_rgb():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 10   # blue channel in BGR
    bgr[..., 2] = 200  # red channel in BGR
    msg = SimpleNamespace(encoding="bgr8", height=2, width=2, data=bgr.tobytes())
    rgb = decode_image_msg(msg)
    assert rgb[0, 0, 0] == 200  # red first in RGB
    assert rgb[0, 0, 2] == 10


def test_decode_compressed_image_canonicalized_to_rgb():
    import cv2
    # compressed_image_transport encodes BGR; a red square in BGR is (0,0,255).
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[..., 2] = 255
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    msg = SimpleNamespace(format="rgb8; jpeg compressed bgr8", data=buf.tobytes())
    decoded = decode_image_msg(msg)
    assert decoded.shape == (8, 8, 3)
    assert decoded[4, 4, 0] > 200  # red first in canonical RGB
    assert decoded[4, 4, 2] < 50


def test_stamp_of_fallback():
    class FakeStamp:
        def __init__(self, s):
            self._s = s
        def to_sec(self):
            return self._s

    stamped = SimpleNamespace(header=SimpleNamespace(stamp=FakeStamp(123.5)))
    assert stamp_of(stamped, fallback=1.0) == 123.5
    zero = SimpleNamespace(header=SimpleNamespace(stamp=FakeStamp(0.0)))
    assert stamp_of(zero, fallback=1.0) == 1.0
    assert stamp_of(SimpleNamespace(), fallback=2.0) == 2.0


# ---------------------------------------------------------------------------
# BagCursor
# ---------------------------------------------------------------------------

def make_cursor():
    return BagCursor({
        "fast": [(t, float(i)) for i, t in enumerate(np.arange(0.0, 1.0, 0.1))],
        "slow": [(0.05, "a"), (0.55, "b")],
    })


def test_causal_zoh():
    cursor = make_cursor()
    out = cursor.advance(0.0)
    assert out["fast"].value == 0.0 and out["fast"].stamp == 0.0
    assert "slow" not in out  # first slow sample is at 0.05

    out = cursor.advance(0.31)
    assert out["fast"].value == 3.0            # latest <= 0.31 is t=0.3
    assert out["slow"].value == "a"            # ZOH from 0.05

    out = cursor.advance(0.56)
    assert out["slow"].value == "b"


def test_zoh_duplicates_between_samples():
    cursor = make_cursor()
    v1 = cursor.advance(0.42)["slow"]
    v2 = cursor.advance(0.48)["slow"]
    assert v1 == v2  # same underlying sample duplicated


def test_advance_requires_nondecreasing_time():
    cursor = make_cursor()
    cursor.advance(0.5)
    with pytest.raises(ValueError):
        cursor.advance(0.4)


def test_seek_allows_rewind():
    cursor = make_cursor()
    cursor.advance(0.9)
    out = cursor.seek(0.11)
    assert out["fast"].value == 1.0
    assert out["slow"].value == "a"


def test_lazy_thunk_decoded_once_per_sample():
    calls = []

    def thunk():
        calls.append(1)
        return "decoded"

    cursor = BagCursor({"img": [(0.0, thunk)]})
    assert cursor.advance(0.0)["img"].value == "decoded"
    assert cursor.advance(0.5)["img"].value == "decoded"
    assert len(calls) == 1  # memoized while the sample index is unchanged


def test_first_last_stamp():
    cursor = make_cursor()
    assert cursor.first_stamp("fast") == 0.0
    assert cursor.last_stamp("slow") == 0.55
