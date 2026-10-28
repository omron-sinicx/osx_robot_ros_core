#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13

SCRIPT_DIR = Path(__file__).resolve().parent

# Default to the force_deformation_data folder written by deformation_measurement.py
# (a sibling of this scripts/ directory), so the analysis runs from the repo with no edits.
BASE_PATH = SCRIPT_DIR.parent / "force_deformation_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "deformation_analysis_output"

# Optional archive of reference (rigid / Fin-Ray) datasets. Disabled unless a path is
# supplied via --archive-path, so the script never depends on data outside the repo.
ARCHIVE_FORCE_DATA_PATH = None

FOLDER_ALIASES = {
    "4_2_haf": "4_2_half",
}

MODE_STYLE_KEYWORDS = [
    ("free", {"label": "CLAW (free)", "color": "#44AA99"}),
    ("half", {"label": "CLAW (half)", "color": "#AA4499"}),
    ("stiff", {"label": "CLAW (stiff)", "color": "#CC6677"}),
]

LINEAR_DIRECTIONS = [
    ("x", "X Translation"),
    ("y", "Y Translation"),
    ("z", "Z Translation"),
    ("xy_pos45", "XY +45°"),
    ("xy_neg45", "XY -45°"),
]

LINEAR_DIRECTION_STYLES = {
    "x": {"label": "X", "linestyle": "solid"},
    "y": {"label": "Y", "linestyle": "dashed"},
    "z": {"label": "Z", "linestyle": "dotted"},
    "xy_pos45": {"label": "XY +45°", "linestyle": "dashdot"},
    "xy_neg45": {"label": "XY -45°", "linestyle": (0, (3, 1, 1, 1))},
}

ROTATIONAL_DIRECTION_COLUMNS = {
    "yaw": ("Orient_2", "Tz"),
    "pitch": ("Orient_3", "Ty"),
    "roll": ("Orient_1", "Tx"),
}

ROLL_FAMILY_DIRECTIONS = [
    ("roll", "Roll", True),
    ("roll_pitch_pos45", "Roll-Pitch +45°", False),
    ("roll_pitch_neg45", "Roll-Pitch -45°", False),
]

ROTATIONAL_DIRECTIONS = [
    ("yaw", "Yaw", True),
    ("pitch", "Pitch", False),
    ("roll", "Roll", True),
    ("roll_pitch_pos45", "Roll-Pitch +45°", False),
    ("roll_pitch_neg45", "Roll-Pitch -45°", False),
]

ROTATIONAL_DIRECTION_STYLES = {
    "yaw": {"label": "Yaw", "linestyle": "solid"},
    "pitch": {"label": "Pitch", "linestyle": "dashed"},
    "roll": {"label": "Roll", "linestyle": "dotted"},
    "roll_pitch_pos45": {"label": "Roll-Pitch +45°", "linestyle": "dashdot"},
    "roll_pitch_neg45": {"label": "Roll-Pitch -45°", "linestyle": (0, (3, 1, 1, 1))},
}

SELECTED_LINEAR_DIRECTIONS = [
    ("xy_pos45", "XY +45°"),
    ("xy_neg45", "XY -45°"),
    ("z", "Z Translation"),
]

SELECTED_ROTATIONAL_DIRECTIONS = [
    ("roll_pitch_pos45", "Roll-Pitch +45°", False),
    ("roll_pitch_neg45", "Roll-Pitch -45°", False),
    ("yaw", "Yaw", True),
]

SELECTED_AXIS_LIMITS = {
    "xy_pos45": {"xlim": (0, 25), "ylim": (0, 15)},
    "xy_neg45": {"xlim": (0, 25), "ylim": (0, 15)},
    "z": {"xlim": (0, 17.5), "ylim": (0, 20)},
    "roll_pitch_pos45": {"xlim": (0, 17.5), "ylim": (0, 1)},
    "roll_pitch_neg45": {"xlim": (0, 17.5), "ylim": (0, 1)},
    "yaw": {"xlim": (0, 30), "ylim": (0, 0.6)},
}

ARCHIVE_REFERENCE_STYLES = {
    "rigid_xy_neg45": {"label": "Rigid X ref", "color": "#6E6E6E", "linestyle": "--"},
    "rigid_z": {"label": "Rigid Z ref", "color": "#6E6E6E", "linestyle": "-."},
    "finray_y_xy_neg45": {"label": "Finray Y ref", "color": "#0B3C5D", "linestyle": "--"},
    "finray_z_30_2_z": {"label": "Finray Z ref", "color": "#0B3C5D", "linestyle": "-."},
}


def configure_plot_style() -> None:
    if sns is not None:
        sns.set(
            style="whitegrid",
            context="paper",
            font_scale=1.0,
            rc={
                "axes.titlesize": TITLE_SIZE,
                "axes.labelsize": LABEL_SIZE,
                "xtick.labelsize": TICK_SIZE,
                "ytick.labelsize": TICK_SIZE,
                "legend.fontsize": LEGEND_SIZE,
                "text.color": "black",
                "axes.labelcolor": "black",
                "axes.titlecolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
            },
        )
    else:  # pragma: no cover
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "axes.titlesize": TITLE_SIZE,
                "axes.labelsize": LABEL_SIZE,
                "xtick.labelsize": TICK_SIZE,
                "ytick.labelsize": TICK_SIZE,
                "legend.fontsize": LEGEND_SIZE,
            }
        )


def resolve_folder(base_path: Path, folder_name: str) -> Path:
    canonical_name = FOLDER_ALIASES.get(folder_name, folder_name)
    return base_path / canonical_name


def infer_mode_style(folder_name: str, color_override: str | None = None) -> dict[str, str]:
    canonical_name = FOLDER_ALIASES.get(folder_name, folder_name)
    lower_name = canonical_name.lower()
    style = {"label": canonical_name, "color": "#1f77b4"}

    for keyword, keyword_style in MODE_STYLE_KEYWORDS:
        if keyword in lower_name:
            style = keyword_style.copy()
            break

    if color_override:
        style["color"] = color_override

    return style


def build_mode_styles(folder_names: list[str], colors: list[str] | None) -> dict[str, dict[str, str]]:
    if colors is not None and len(colors) != len(folder_names):
        raise ValueError("--colors must have the same number of entries as --folders.")

    mode_styles: dict[str, dict[str, str]] = {}
    for index, folder_name in enumerate(folder_names):
        color_override = colors[index] if colors is not None else None
        mode_styles[folder_name] = infer_mode_style(folder_name, color_override=color_override)

    return mode_styles


def detect_axis_from_filename(filename: str) -> str | None:
    lower_name = filename.lower()
    detection_order = [
        ("roll_pitch_pos45", "roll_pitch_pos45"),
        ("pos45", "xy_pos45"),
        ("roll_pitch_neg45", "roll_pitch_neg45"),
        ("neg45", "xy_neg45"),
        ("pitch", "pitch"),
        ("roll", "roll"),
        ("yaw", "yaw"),
        ("_x_", "x"),
        ("_y_", "y"),
        ("_z_", "z"),
    ]

    for pattern, axis in detection_order:
        if pattern in lower_name:
            if pattern == "pos45" and "roll_pitch_pos45" in lower_name:
                continue
            if pattern == "neg45" and "roll_pitch_neg45" in lower_name:
                continue
            return axis
    return None


def load_dataset_repetitions(root_path: Path, axis: str | None = None) -> list[pd.DataFrame]:
    if axis:
        pattern = f"deformation_data_{axis}_rep*_*.csv"
    else:
        pattern = "deformation_data_*_rep*_*.csv"

    files = sorted(glob.glob(str(root_path / pattern)))
    repetitions: list[pd.DataFrame] = []

    print(f"Found {len(files)} files in {root_path}")
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            df["detected_axis"] = detect_axis_from_filename(filename)
            repetitions.append(df)
            print(f"  Loaded: {filename} ({len(df)} rows, axis={df['detected_axis'].iloc[0]})")
        except Exception as exc:  # pragma: no cover
            print(f"  Error loading {file_path}: {exc}")

    return repetitions


def normalize_initial_position(data: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    data_normalized = data.copy()
    if not data.empty:
        data_normalized[axis_col] = data[axis_col] - data[axis_col].iloc[0]
    return data_normalized


def process_linear_dataset(root_path: Path, axis: str) -> list[dict[str, np.ndarray]]:
    repetitions = load_dataset_repetitions(root_path, axis)
    processed_reps: list[dict[str, np.ndarray]] = []

    linear_columns = {
        "x": ("Pos_X", "Fx"),
        "y": ("Pos_Y", "Fy"),
        "z": ("Pos_Z", "Fz"),
    }

    for rep_data in repetitions:
        if axis in {"xy_pos45", "xy_neg45"}:
            x_pos = rep_data["Pos_X"].to_numpy()
            y_pos = rep_data["Pos_Y"].to_numpy()
            x_normalized = x_pos - x_pos[0]
            y_normalized = y_pos - y_pos[0]
            displacement = np.sqrt(x_normalized**2 + y_normalized**2)

            fx = rep_data["Fx"].to_numpy()
            fy = rep_data["Fy"].to_numpy()
            force = -np.sqrt(fx**2 + fy**2)
        else:
            position_col, force_col = linear_columns[axis]
            rep_normalized = normalize_initial_position(rep_data, position_col)
            displacement = rep_normalized[position_col].to_numpy()
            force = rep_normalized[force_col].to_numpy()

        processed_reps.append(
            {
                "displacement": displacement,
                "force": force,
            }
        )

    return processed_reps


def compute_loading_stats(
    processed_reps: list[dict[str, np.ndarray]], n_points: int = 1000
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not processed_reps:
        return None, None, None

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for rep in processed_reps:
        displacement = rep["displacement"]
        force = rep["force"]

        if len(displacement) == 0:
            continue

        max_force_idx = int(np.argmax(np.abs(force)))
        displacement = displacement[: max_force_idx + 1]
        force = force[: max_force_idx + 1]

        order = np.argsort(displacement)
        xs.append(displacement[order])
        ys.append(force[order])

    if not xs:
        return None, None, None

    xmin = max(x.min() for x in xs)
    xmax = min(x.max() for x in xs)
    if xmin >= xmax:
        xmin = min(x.min() for x in xs)
        xmax = max(x.max() for x in xs)
    common_x = np.linspace(xmin, xmax, n_points)

    y_interp = []
    for x, y in zip(xs, ys):
        try:
            y_interp.append(np.interp(common_x, x, y, left=np.nan, right=np.nan))
        except Exception as exc:  # pragma: no cover
            print(f"  Interpolation warning: {exc}")

    if not y_interp:
        return None, None, None

    y_array = np.vstack(y_interp)
    valid_columns = ~np.all(np.isnan(y_array), axis=0)
    if not np.any(valid_columns):
        return None, None, None

    y_array = y_array[:, valid_columns]
    common_x = common_x[valid_columns]
    y_array = y_array - y_array[:, 0:1]

    mean_force = np.nanmean(y_array, axis=0)
    std_force = np.nanstd(y_array, axis=0)
    x_mm = (common_x - common_x.min()) * 1000.0

    if np.all(std_force == 0) or np.all(np.isnan(std_force)) or np.nanmax(std_force) < 0.01:
        std_force = np.abs(mean_force) * 0.05

    return x_mm, mean_force, std_force


def compute_rotational_raw_stats(
    processed_reps: list[dict[str, np.ndarray]], n_points_cap: int = 1000
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not processed_reps:
        return None, None, None

    displacement_data = []
    force_data = []

    for rep in processed_reps:
        displacement = rep["displacement"]
        force = rep["force"]
        if len(displacement) == 0:
            continue

        force = force - force[0]
        displacement_deg = np.rad2deg(displacement)

        displacement_data.append(displacement_deg)
        force_data.append(force)

    if not displacement_data:
        return None, None, None

    min_displacement = max(np.min(disp) for disp in displacement_data)
    max_displacement = min(np.max(disp) for disp in displacement_data)
    if min_displacement >= max_displacement:
        min_displacement = min(np.min(disp) for disp in displacement_data)
        max_displacement = max(np.max(disp) for disp in displacement_data)
    max_length = max(len(disp) for disp in displacement_data)
    n_points = min(max_length, n_points_cap)

    common_displacement = np.linspace(min_displacement, max_displacement, n_points)

    interpolated_forces = []
    for disp, force in zip(displacement_data, force_data):
        if len(disp) < 2:
            continue
        sort_idx = np.argsort(disp)
        disp_sorted = np.asarray(disp)[sort_idx]
        force_sorted = np.asarray(force)[sort_idx]
        interpolated = np.interp(common_displacement, disp_sorted, force_sorted, left=np.nan, right=np.nan)
        interpolated_forces.append(interpolated)

    if not interpolated_forces:
        return None, None, None

    force_array = np.asarray(interpolated_forces)
    mean_force = np.nanmean(force_array, axis=0)
    std_force = np.nanstd(force_array, axis=0)
    valid_mask = ~(np.isnan(mean_force) | np.isnan(std_force))

    if not np.any(valid_mask):
        return None, None, None

    mean_displacement = common_displacement[valid_mask]
    mean_force = mean_force[valid_mask]
    std_force = std_force[valid_mask]
    mean_force = mean_force - mean_force[0]

    if np.all(std_force == 0) or np.all(np.isnan(std_force)) or np.nanmax(std_force) < 0.01:
        std_force = np.abs(mean_force) * 0.05

    return mean_displacement, mean_force, std_force


def process_rotational_datasets(folder_names: list[str], base_path: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    datasets = {
        "yaw": {},
        "pitch": {},
        "roll": {},
        "roll_pitch_pos45": {},
        "roll_pitch_neg45": {},
    }

    for folder_name in folder_names:
        root_path = resolve_folder(base_path, folder_name)
        repetitions = load_dataset_repetitions(root_path, axis=None)
        grouped_reps = {key: [] for key in datasets}

        for rep_data in repetitions:
            axis = rep_data["detected_axis"].iloc[0] if "detected_axis" in rep_data.columns else None
            if axis in grouped_reps:
                grouped_reps[axis].append(rep_data)

        for axis in ("yaw", "pitch", "roll"):
            axis_reps = []
            position_col, torque_col = ROTATIONAL_DIRECTION_COLUMNS[axis]
            for rep_data in grouped_reps[axis]:
                if axis == "yaw":
                    min_torque_idx = int(np.argmin(rep_data[torque_col].to_numpy()))
                    rep_data = rep_data.iloc[: min_torque_idx + 1].copy()
                elif axis == "roll":
                    torque_magnitude = np.sqrt(rep_data["Tx"] ** 2 + rep_data["Ty"] ** 2 + rep_data["Tz"] ** 2)
                    max_torque_idx = int(np.argmax(torque_magnitude))
                    rep_data = rep_data.iloc[: max_torque_idx + 1].copy()

                rep_normalized = normalize_initial_position(rep_data, position_col)
                displacement = rep_normalized[position_col].to_numpy()
                force = rep_normalized[torque_col].to_numpy()
                force = force - force[0]

                axis_reps.append({"displacement": displacement, "force": force})

            displacement, mean_force, std_force = compute_rotational_raw_stats(axis_reps)
            if displacement is not None:
                datasets[axis][folder_name] = {
                    "displacement": displacement,
                    "force": mean_force,
                    "force_std": std_force,
                    "n_reps": len(axis_reps),
                }

        for axis in ("roll_pitch_pos45", "roll_pitch_neg45"):
            axis_reps = []
            for rep_data in grouped_reps[axis]:
                torque_magnitude = np.sqrt(rep_data["Tx"] ** 2 + rep_data["Ty"] ** 2 + rep_data["Tz"] ** 2)
                max_torque_idx = int(np.argmax(torque_magnitude))
                rep_truncated = rep_data.iloc[: max_torque_idx + 1].copy()

                initial_orient = rep_truncated[["Orient_1", "Orient_2", "Orient_3"]].iloc[0].to_numpy()
                current_orient = rep_truncated[["Orient_1", "Orient_2", "Orient_3"]].to_numpy()
                orient_change = current_orient - initial_orient
                displacement = np.linalg.norm(orient_change, axis=1)

                torque_components = rep_truncated[["Tx", "Ty", "Tz"]].to_numpy()
                force = np.linalg.norm(torque_components, axis=1)
                force = force - force[0]

                axis_reps.append({"displacement": displacement, "force": force})

            displacement, mean_force, std_force = compute_rotational_raw_stats(axis_reps)
            if displacement is not None:
                datasets[axis][folder_name] = {
                    "displacement": displacement,
                    "force": mean_force,
                    "force_std": std_force,
                    "n_reps": len(axis_reps),
                }

    return datasets


def create_linear_comparison_plot(
    folder_names: list[str], base_path: Path, output_dir: Path, mode_styles: dict[str, dict[str, str]]
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    mode_handles = []
    mode_labels = []

    for axis_index, (direction, title) in enumerate(LINEAR_DIRECTIONS):
        ax = axes[axis_index]
        for folder_name in folder_names:
            root_path = resolve_folder(base_path, folder_name)
            processed_reps = process_linear_dataset(root_path, direction)
            displacement, mean_force, std_force = compute_loading_stats(processed_reps)
            if displacement is None:
                continue

            mode_style = mode_styles[folder_name]
            line, = ax.plot(
                displacement,
                -mean_force,
                color=mode_style["color"],
                linestyle="solid",
                linewidth=2.8,
            )
            ax.fill_between(
                displacement,
                -mean_force - std_force,
                -mean_force + std_force,
                color=mode_style["color"],
                alpha=0.08,
                edgecolor="none",
            )

            if axis_index == 0:
                mode_handles.append(line)
                mode_labels.append(mode_style["label"])

        ax.set_title(title)
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Force (N)")
        axis_limits = SELECTED_AXIS_LIMITS.get(direction)
        if axis_limits is not None:
            ax.set_xlim(*axis_limits["xlim"])
            ax.set_ylim(*axis_limits["ylim"])
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("4/2 Translational Force-Displacement Comparison", fontsize=20, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.legend(mode_handles, mode_labels, loc="lower center", ncol=max(1, len(mode_labels)), title="Mode")

    output_path = output_dir / "linear_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_rotational_comparison_plot(
    datasets: dict[str, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
    mode_styles: dict[str, dict[str, str]],
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    mode_handles = []
    mode_labels = []

    for axis_index, (direction, title, reverse_sign) in enumerate(ROTATIONAL_DIRECTIONS):
        ax = axes[axis_index]
        for folder_name, data in datasets.get(direction, {}).items():
            mode_style = mode_styles[folder_name]
            displacement = data["displacement"]
            torque = -data["force"] if reverse_sign else data["force"]
            torque_std = data["force_std"]

            line, = ax.plot(
                displacement,
                torque,
                color=mode_style["color"],
                linestyle="solid",
                linewidth=3.0,
            )
            ax.fill_between(
                displacement,
                torque - torque_std,
                torque + torque_std,
                color=mode_style["color"],
                alpha=0.08,
                edgecolor="none",
            )

            if axis_index == 0:
                mode_handles.append(line)
                mode_labels.append(mode_style["label"])

        ax.set_title(title)
        ax.set_xlabel("Angular Displacement (degrees)")
        ax.set_ylabel("Torque (Nm)")
        axis_limits = SELECTED_AXIS_LIMITS.get(direction)
        if axis_limits is not None:
            ax.set_xlim(*axis_limits["xlim"])
            ax.set_ylim(*axis_limits["ylim"])
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("4/2 Rotational Torque Comparison", fontsize=20, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.legend(mode_handles, mode_labels, loc="lower center", ncol=max(1, len(mode_labels)), title="Mode")

    output_path = output_dir / "rotational_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_archive_reference_curves(archive_path: Path) -> dict[str, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]:
    archive_specs = {
        "rigid_xy_neg45": (archive_path / "rigid", "x"),
        "rigid_z": (archive_path / "rigid", "z"),
        "finray_y_xy_neg45": (archive_path / "finray_y", "y"),
        "finray_z_30_2_z": (archive_path / "finray_z_30_2", "z"),
    }

    curves: dict[str, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]] = {}
    for key, (root_path, axis) in archive_specs.items():
        processed_reps = process_linear_dataset(root_path, axis)
        curves[key] = compute_loading_stats(processed_reps)
    return curves


def create_selected_comparison_plot(
    folder_names: list[str],
    base_path: Path,
    rotational_datasets: dict[str, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
    mode_styles: dict[str, dict[str, str]],
    include_archive_refs: bool = False,
    archive_path: Path | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    archive_curves = (
        load_archive_reference_curves(archive_path)
        if include_archive_refs and archive_path is not None
        else {}
    )

    for axis_index, (direction, title) in enumerate(SELECTED_LINEAR_DIRECTIONS):
        ax = axes[axis_index]
        for folder_name in folder_names:
            root_path = resolve_folder(base_path, folder_name)
            processed_reps = process_linear_dataset(root_path, direction)
            displacement, mean_force, std_force = compute_loading_stats(processed_reps)
            if displacement is None:
                continue

            mode_style = mode_styles[folder_name]
            ax.plot(
                displacement,
                -mean_force,
                color=mode_style["color"],
                linestyle="solid",
                linewidth=2.8,
            )
            ax.fill_between(
                displacement,
                -mean_force - std_force,
                -mean_force + std_force,
                color=mode_style["color"],
                alpha=0.08,
                edgecolor="none",
            )

        if include_archive_refs and direction == "xy_neg45":
            for key in ("rigid_xy_neg45", "finray_y_xy_neg45"):
                displacement, mean_force, _ = archive_curves.get(key, (None, None, None))
                if displacement is None:
                    continue
                style = ARCHIVE_REFERENCE_STYLES[key]
                ax.plot(
                    displacement,
                    -mean_force,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.6,
                )

        if include_archive_refs and direction == "z":
            for key in ("rigid_z", "finray_z_30_2_z"):
                displacement, mean_force, _ = archive_curves.get(key, (None, None, None))
                if displacement is None:
                    continue
                style = ARCHIVE_REFERENCE_STYLES[key]
                ax.plot(
                    displacement,
                    -mean_force,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.6,
                )

        ax.set_title(title)
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Force (N)")
        axis_limits = SELECTED_AXIS_LIMITS.get(direction)
        if axis_limits is not None:
            ax.set_xlim(*axis_limits["xlim"])
            ax.set_ylim(*axis_limits["ylim"])
        ax.grid(True, alpha=0.3)

    for offset, (direction, title, reverse_sign) in enumerate(SELECTED_ROTATIONAL_DIRECTIONS, start=3):
        ax = axes[offset]
        for folder_name, data in rotational_datasets.get(direction, {}).items():
            mode_style = mode_styles[folder_name]
            displacement = data["displacement"]
            torque = -data["force"] if reverse_sign else data["force"]
            torque_std = data["force_std"]

            ax.plot(
                displacement,
                torque,
                color=mode_style["color"],
                linestyle="solid",
                linewidth=3.0,
            )
            ax.fill_between(
                displacement,
                torque - torque_std,
                torque + torque_std,
                color=mode_style["color"],
                alpha=0.08,
                edgecolor="none",
            )

        ax.set_title(title)
        ax.set_xlabel("Angular Displacement (degrees)")
        ax.set_ylabel("Torque (Nm)")
        axis_limits = SELECTED_AXIS_LIMITS.get(direction)
        if axis_limits is not None:
            ax.set_xlim(*axis_limits["xlim"])
            ax.set_ylim(*axis_limits["ylim"])
        ax.grid(True, alpha=0.3)

    legend_handles = []
    legend_labels = []
    for folder_name in folder_names:
        style = mode_styles[folder_name]
        legend_handles.append(Line2D([0], [0], color=style["color"], linewidth=3.0))
        legend_labels.append(style["label"])

    if include_archive_refs:
        for key in ("rigid_xy_neg45", "rigid_z", "finray_y_xy_neg45", "finray_z_30_2_z"):
            style = ARCHIVE_REFERENCE_STYLES[key]
            legend_handles.append(
                Line2D([0], [0], color=style["color"], linewidth=2.6, linestyle=style["linestyle"])
            )
            legend_labels.append(style["label"])

    title = "Focused Comparison with Archive References" if include_archive_refs else "Focused Comparison"
    legend_title = "Mode / Reference" if include_archive_refs else "Mode"
    output_name = "selected_comparison_with_archive.png" if include_archive_refs else "selected_comparison.png"

    fig.suptitle(title, fontsize=20, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        title=legend_title,
    )

    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 4/2 force-deformation datasets.")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=BASE_PATH,
        help="Directory containing force deformation dataset folders.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["4_6_free_45_3", "4_6_half_45_3", "4_6_stiff_45_3"],
        help="Dataset folders to compare. '4_2_haf' is accepted as an alias for '4_2_half'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="Optional colors aligned with --folders, e.g. --colors '#44AA99' '#AA4499' '#CC6677'.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=ARCHIVE_FORCE_DATA_PATH,
        help="Optional directory of reference (rigid / Fin-Ray) datasets. Enables the "
             "archive-reference comparison plot when provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot_style()

    base_path = args.base_path
    output_dir = args.output_dir
    folder_names = [FOLDER_ALIASES.get(folder, folder) for folder in args.folders]
    mode_styles = build_mode_styles(folder_names, args.colors)

    # Only draw the archive-reference plot when a real archive directory is available.
    archive_path = args.archive_path
    include_archive_refs = archive_path is not None and archive_path.exists()
    if archive_path is not None and not include_archive_refs:
        print(f"Archive path {archive_path} not found; skipping archive-reference plot.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 4/2 FORCE-DEFORMATION ANALYSIS ===")
    print(f"Base path: {base_path}")
    print(f"Folders: {folder_names}")
    print(f"Mode styles: {mode_styles}")
    print(f"Output dir: {output_dir}")

    saved_files = []
    saved_files.append(create_linear_comparison_plot(folder_names, base_path, output_dir, mode_styles))

    rotational_datasets = process_rotational_datasets(folder_names, base_path)
    saved_files.append(create_rotational_comparison_plot(rotational_datasets, output_dir, mode_styles))
    saved_files.append(
        create_selected_comparison_plot(
            folder_names, base_path, rotational_datasets, output_dir, mode_styles, include_archive_refs=False
        )
    )
    if include_archive_refs:
        saved_files.append(
            create_selected_comparison_plot(
                folder_names, base_path, rotational_datasets, output_dir, mode_styles,
                include_archive_refs=True, archive_path=archive_path,
            )
        )

    print("\nSaved plots:")
    for output_file in saved_files:
        print(f"- {output_file}")


if __name__ == "__main__":
    main()