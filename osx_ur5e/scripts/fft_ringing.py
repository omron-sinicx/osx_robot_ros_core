#!/usr/bin/env python3
"""FFT of the post-impact ringing, to identify the structural resonance.

Why two spectra: the FT 300-S samples at 100 Hz (Nyquist 50 Hz), so any
resonance above 50 Hz aliases in the FT signal. The robot built-in wrench runs
at ~500 Hz (Nyquist 250 Hz), so it reveals the *true* ringing frequency. We FFT
both post-tap windows and compare.

Uses a signed force component (the dominant impact axis), NOT the magnitude:
rectifying |f| would double/distort the apparent frequency.

    python3 fft_ringing.py ft_delta.bag
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
    """Per-axis peak |deviation from baseline|. Baseline = median (robust to the
    brief impact spikes). Use this, not max-min: a one-sided impact axis (Fz here)
    has a small max-min but a large deviation, whereas a symmetrically-ringing
    lateral axis inflates max-min -- so max-min under-ranks the true impact axis."""
    return np.abs(f - np.median(f, axis=0)).max(axis=0)


def dominant_axis(f):
    """Index of the force component with the largest peak deviation from baseline."""
    return int(np.argmax(peak_deviation(f)))


def avg_spectrum(t, x, taps, fs, win_len, skip):
    """Average |FFT| of post-tap windows. Returns (freqs, mag, example_window)."""
    nwin = int(win_len * fs)
    win = np.hanning(nwin)
    acc = None
    example = None
    for tp in taps:
        t0 = tp + skip
        grid = t0 + np.arange(nwin) / fs
        if grid[-1] > t[-1]:
            continue
        seg = np.interp(grid, t, x)
        seg = seg - seg.mean()
        if example is None:
            example = (grid - tp, seg.copy())
        spec = np.abs(np.fft.rfft(seg * win))
        acc = spec if acc is None else acc + spec
    freqs = np.fft.rfftfreq(nwin, 1.0 / fs)
    return freqs, acc, example


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--ft-topic", default="/robotiq_ft_wrench")
    ap.add_argument("--robot-topic", default="/wrench")
    ap.add_argument("--win-len", type=float, default=0.25, help="post-tap window [s]")
    ap.add_argument("--skip", type=float, default=0.002, help="skip after peak [s]")
    ap.add_argument("--min-sep", type=float, default=0.4)
    ap.add_argument("--height-sigma", type=float, default=6.0)
    ap.add_argument("--ft-axis", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--robot-axis", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("-o", "--out", default="ft_fft.png")
    args = ap.parse_args()

    bag = rosbag.Bag(args.bag)
    t_ft, f_ft = load_forces(bag, args.ft_topic)
    t_rb, f_rb = load_forces(bag, args.robot_topic)
    bag.close()
    if len(t_ft) == 0 or len(t_rb) == 0:
        raise SystemExit("empty topic(s)")

    fs_ft = len(t_ft) / (t_ft[-1] - t_ft[0])
    fs_rb = len(t_rb) / (t_rb[-1] - t_rb[0])

    # detect taps on robot magnitude
    mag_rb = np.linalg.norm(f_rb, axis=1)
    thr = mag_rb.mean() + args.height_sigma * mag_rb.std()
    pk, _ = find_peaks(mag_rb, height=thr, distance=int(args.min_sep * fs_rb))
    taps = t_rb[pk]
    if len(taps) == 0:
        raise SystemExit("no taps detected")

    names = "xyz"
    dev_ft = peak_deviation(f_ft)
    dev_rb = peak_deviation(f_rb)
    ax_ft = dominant_axis(f_ft) if args.ft_axis == "auto" else names.index(args.ft_axis)
    ax_rb = dominant_axis(f_rb) if args.robot_axis == "auto" else names.index(args.robot_axis)
    print("FT rate ~%.1f Hz (Nyquist %.0f Hz) | robot rate ~%.1f Hz (Nyquist %.0f Hz)"
          % (fs_ft, fs_ft / 2, fs_rb, fs_rb / 2))
    print("FT    peak deviation [N]: Fx=%.1f Fy=%.1f Fz=%.1f" % tuple(dev_ft))
    print("robot peak deviation [N]: Fx=%.1f Fy=%.1f Fz=%.1f" % tuple(dev_rb))
    print("using axis: FT=F%s  robot=F%s | %d taps" % (names[ax_ft], names[ax_rb], len(taps)))

    fr_rb, sp_rb, ex_rb = avg_spectrum(t_rb, f_rb[:, ax_rb], taps, fs_rb, args.win_len, args.skip)
    fr_ft, sp_ft, ex_ft = avg_spectrum(t_ft, f_ft[:, ax_ft], taps, fs_ft, args.win_len, args.skip)

    def peak_freq(fr, sp, fmin=12.0):
        # ignore the < fmin region: a damped pulse has a large low-freq envelope
        # there that is NOT a resonance; the structural modes sit above it.
        m = fr >= fmin
        i = np.argmax(sp[m])
        return fr[m][i]

    f_peak_rb = peak_freq(fr_rb, sp_rb)
    f_peak_ft = peak_freq(fr_ft, sp_ft)
    print("dominant ringing freq:  robot = %.1f Hz   FT = %.1f Hz" % (f_peak_rb, f_peak_ft))
    if f_peak_rb > fs_ft / 2:
        alias = abs(f_peak_rb - round(f_peak_rb / fs_ft) * fs_ft)
        print("NOTE: robot resonance %.1f Hz is ABOVE FT Nyquist %.0f Hz -> aliases to ~%.1f Hz in FT"
              % (f_peak_rb, fs_ft / 2, alias))

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(ex_rb[0] * 1000, ex_rb[1], ".-", label="robot F%s" % names[ax_rb])
    ax[0].plot(ex_ft[0] * 1000, ex_ft[1], "o-", ms=4, label="FT F%s" % names[ax_ft], alpha=0.8)
    ax[0].set_title("example post-tap window (time domain)")
    ax[0].set_xlabel("time rel. tap [ms]"); ax[0].legend()

    ax[1].plot(fr_rb, sp_rb)
    ax[1].axvline(f_peak_rb, color="r", ls="--", label="peak %.1f Hz" % f_peak_rb)
    ax[1].axvline(fs_ft / 2, color="g", ls=":", label="FT Nyquist %.0f Hz" % (fs_ft / 2))
    ax[1].set_title("ROBOT spectrum (true, up to %.0f Hz)" % (fs_rb / 2))
    ax[1].set_xlabel("Hz"); ax[1].set_xlim(0, fs_rb / 2); ax[1].legend()

    ax[2].plot(fr_ft, sp_ft)
    ax[2].axvline(f_peak_ft, color="r", ls="--", label="peak %.1f Hz" % f_peak_ft)
    ax[2].set_title("FT spectrum (limited to %.0f Hz Nyquist)" % (fs_ft / 2))
    ax[2].set_xlabel("Hz"); ax[2].set_xlim(0, fs_ft / 2); ax[2].legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved plot: %s" % args.out)


if __name__ == "__main__":
    main()
