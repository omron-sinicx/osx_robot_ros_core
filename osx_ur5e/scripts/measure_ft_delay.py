#!/usr/bin/env python3
"""Estimate the time offset (delta) between the external Robotiq FT 300-S stream
(read over USB on the PC) and the robot's built-in wrench (RTDE pipeline).

    delta = t_FT - t_robot   (positive => FT is later than the robot)

Method (rationale)
------------------
We measure the per-tap ONSET (rising-edge) time difference on a SINGLE signed
axis (Fz by default), NOT the magnitude |f| and NOT the peak time:

  * |f| rectifies the signal and mixes all axes -> distorts the transient when
    the sensor rings or has a static (un-zeroed) offset.
  * the FT response RINGS, so the peak lands on a ring crest ~15-25 ms late and
    over-estimates the delay. The onset (first departure from baseline) is the
    physically meaningful arrival time.

The FT samples at 100 Hz, so each onset is quantized to ~10 ms; the per-tap
spread is therefore dominated by that quantization (uniform-dist std = 10/sqrt(12)
~= 2.9 ms) plus jitter. Averaging N taps shrinks the *mean's* standard error by
sqrt(N) (random tap phase acts as dither), so the mean delta is resolvable well
below 10 ms even though single samples are not.

    rosbag record -O ft_delta.bag /robotiq_ft_wrench /wrench   # tap a few times
    python3 measure_ft_delay.py ft_delta.bag

Apply the result in the time-shift republisher:
    ft_msg.header.stamp -= rospy.Duration(delta)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import rosbag


def load_forces(bag, topic):
    """Return t[s] and signed force array (N,3) = (Fx,Fy,Fz)."""
    t, f = [], []
    for _, msg, _ in bag.read_messages(topics=[topic]):
        w = msg.wrench.force
        t.append(msg.header.stamp.to_sec())
        f.append((w.x, w.y, w.z))
    return np.asarray(t), np.asarray(f)


def peak_deviation(f):
    """Per-axis peak |deviation from baseline| (baseline = median, robust to the
    brief impact spikes). Use this, not max-min: a one-sided impact axis has a
    small max-min but a large deviation, while a symmetrically-ringing axis
    inflates max-min, so max-min under-ranks the true impact axis."""
    return np.abs(f - np.median(f, axis=0)).max(axis=0)


def onset_time(t, dev, center, half, frac):
    """First time within [center-half, center+half] where |dev| exceeds
    frac * (window peak |dev|). Returns None if the window is empty."""
    sel = (t >= center - half) & (t <= center + half)
    if not np.any(sel):
        return None
    ts, ds = t[sel], np.abs(dev[sel])
    over = np.where(ds > frac * ds.max())[0]
    return ts[over[0]] if len(over) else None


def peak_time(t, dev, center, half):
    sel = (t >= center - half) & (t <= center + half)
    ts, ds = t[sel], np.abs(dev[sel])
    return ts[int(np.argmax(ds))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--ft-topic", default="/robotiq_ft_wrench")
    ap.add_argument("--robot-topic", default="/wrench")
    ap.add_argument("--axis", default="z", choices=["x", "y", "z", "auto"],
                    help="force axis used for onset timing (default z = impact axis)")
    ap.add_argument("--half", type=float, default=0.05, help="per-tap half-window [s]")
    ap.add_argument("--frac", type=float, default=0.25, help="onset threshold as fraction of tap peak")
    ap.add_argument("--min-sep", type=float, default=0.4, help="min tap separation [s]")
    ap.add_argument("--height-sigma", type=float, default=6.0,
                    help="tap detection threshold = sigma*std on robot axis deviation")
    ap.add_argument("-o", "--out", default="ft_delay.png")
    args = ap.parse_args()

    bag = rosbag.Bag(args.bag)
    t_ft, f_ft = load_forces(bag, args.ft_topic)
    t_rb, f_rb = load_forces(bag, args.robot_topic)
    bag.close()
    if len(t_ft) == 0 or len(t_rb) == 0:
        raise SystemExit("empty topic(s): FT=%d robot=%d (was the robot driver running?)"
                         % (len(t_ft), len(t_rb)))

    names = "xyz"
    ax = names.index(args.axis) if args.axis != "auto" else int(np.argmax(peak_deviation(f_rb)))
    fs_ft = len(t_ft) / (t_ft[-1] - t_ft[0])
    fs_rb = len(t_rb) / (t_rb[-1] - t_rb[0])

    # signed deviation from the (robust) baseline on the chosen axis
    dev_ft = f_ft[:, ax] - np.median(f_ft[:, ax])
    dev_rb = f_rb[:, ax] - np.median(f_rb[:, ax])

    # detect taps on the robot axis (sharp, high-rate)
    thr = args.height_sigma * np.std(dev_rb)
    pk, _ = find_peaks(np.abs(dev_rb), height=thr, distance=int(args.min_sep * fs_rb))
    if len(pk) == 0:
        raise SystemExit("no taps detected; lower --height-sigma or check the data")

    print("axis=F%s | FT ~%.0f Hz (%.1f ms grid) | robot ~%.0f Hz | %d taps"
          % (names[ax], fs_ft, 1000.0 / fs_ft, fs_rb, len(pk)))
    print("\n tap |  onset dt [ms] |  peak dt [ms]")
    print("-----+---------------+--------------")
    onsets, peaks = [], []
    for n, ip in enumerate(pk):
        c = t_rb[ip]
        o_rb = onset_time(t_rb, dev_rb, c, args.half, args.frac)
        o_ft = onset_time(t_ft, dev_ft, c, args.half, args.frac)
        p_rb = peak_time(t_rb, dev_rb, c, args.half)
        p_ft = peak_time(t_ft, dev_ft, c, args.half)
        od = (o_ft - o_rb) * 1000 if (o_rb and o_ft) else np.nan
        pd = (p_ft - p_rb) * 1000
        onsets.append(od); peaks.append(pd)
        print("  %2d |   %+8.1f    |   %+7.1f" % (n, od, pd))

    onsets = np.asarray(onsets); peaks = np.asarray(peaks)
    on = onsets[~np.isnan(onsets)]
    mean, pop_std = on.mean(), on.std()
    se = pop_std / np.sqrt(len(on))
    q_std = (1000.0 / fs_ft) / np.sqrt(12)  # quantization-only std floor

    print("\n--- ONSET-based delay (use this) ---")
    print("mean delta   = %+.2f ms" % mean)
    print("std error    =  %.2f ms  (= pop_std/sqrt(N), shrinks with more taps)" % se)
    print("pop. std     =  %.2f ms  (per-sample spread; floor ~%.1f ms from 100 Hz quantization)"
          % (pop_std, q_std))
    print("peak-based   = %+.2f ms  (biased late by ringing; for reference only)" % np.nanmean(peaks))
    print("\n=> republisher:  ft_msg.header.stamp -= rospy.Duration(%.3f)" % (mean / 1000.0))

    # per-tap onset-aligned plot
    ncol = min(3, len(pk)); nrow = int(np.ceil(len(pk) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow), squeeze=False)
    for n, ip in enumerate(pk):
        a = axes[n // ncol][n % ncol]
        c = t_rb[ip]
        for t_s, d_s, lab, mk in ((t_rb, dev_rb, "robot", ".-"), (t_ft, dev_ft, "FT", "o-")):
            sel = (t_s >= c - args.half) & (t_s <= c + args.half)
            d = d_s[sel]
            a.plot((t_s[sel] - c) * 1000, d / (np.abs(d).max() + 1e-9), mk, ms=4, label=lab)
        a.axvline(0, color="k", lw=0.5)
        a.set_title("tap %d  (onset dt=%+.0f ms)" % (n, onsets[n]))
        a.set_xlabel("time rel. robot peak [ms]")
        if n == 0:
            a.legend(fontsize=8)
    for k in range(len(pk), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("F%s onset alignment | mean delta = %+.1f +/- %.1f ms (SE)" % (names[ax], mean, se))
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved plot: %s" % args.out)


if __name__ == "__main__":
    main()
