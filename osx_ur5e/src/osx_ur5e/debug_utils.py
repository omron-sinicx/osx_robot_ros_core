
import logging
from pathlib import Path

import pandas as pd
import torch

from rich.console import Console

from comet.scripts.utils.visualize_episode import plot_factored_from_arrays, plot_virtual_from_arrays

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Per-horizon prediction logger
# ---------------------------------------------------------------------------

def save_horizon_html(horizon_dict: dict, call_num: int, save_dir: Path):
    """Save the full predicted action horizon as an interactive HTML file.

    Auto-detects factored vs virtual displacement representation and produces
    the same 3D visualization style as visualize_episode.py (reference trajectory,
    contact directions, reconstructed virtual target, force/stiffness panels).

    Args:
        horizon_dict: {feature_name: np.ndarray [horizon, dim]} in physical units.
        call_num: Prediction call index.
        save_dir: Directory to write HTML files into.
    """
    if "action.contact_direction" in horizon_dict:
        fig = plot_factored_from_arrays(
            ref_positions=horizon_dict["action.ref_position"][:, :3],
            contact_directions=horizon_dict["action.contact_direction"],
            normal_forces=horizon_dict["action.normal_force"].flatten(),
            stiffnesses=horizon_dict["action.estimated_stiffness"].flatten(),
            title=f"Predicted Horizon {call_num} — Factored",
        )
    elif "action.virtual_target_position" in horizon_dict:
        ref_pos = horizon_dict.get("action.ref_position")
        fig = plot_virtual_from_arrays(
            vt_positions=horizon_dict["action.virtual_target_position"][:, :3],
            stiffnesses=horizon_dict["action.estimated_stiffness"].flatten(),
            ref_positions=ref_pos[:, :3] if ref_pos is not None else None,
            title=f"Predicted Horizon {call_num} — Virtual Displacement",
        )
    else:
        fig = _plot_horizon_generic(horizon_dict, call_num)

    fig.write_html(save_dir / f"horizon_{call_num:04d}.html")

    csv_data = {}
    for key, arr in horizon_dict.items():
        if arr.ndim == 1:
            csv_data[key] = arr
        else:
            for d in range(arr.shape[1]):
                csv_data[f"{key}[{d}]"] = arr[:, d]
    pd.DataFrame(csv_data).to_csv(
        save_dir / f"horizon_{call_num:04d}.csv", index_label="step")


def _plot_horizon_generic(horizon_dict: dict, call_num: int) -> go.Figure:
    """Fallback: per-feature time-series plot for unknown action representations."""
    feature_names = list(horizon_dict.keys())
    n_panels = len(feature_names)

    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=feature_names,
        vertical_spacing=0.06,
    )

    for idx, key in enumerate(feature_names):
        values = horizon_dict[key]
        if values.ndim == 1:
            values = values[:, None]
        T, D = values.shape
        t_axis = list(range(T))

        for d in range(D):
            fig.add_trace(go.Scatter(
                x=t_axis, y=values[:, d],
                mode="lines+markers",
                name=f"{key.split('.')[-1]}[{d}]",
                legendgroup=key,
            ), row=idx + 1, col=1)

        fig.update_yaxes(title_text=key.split(".")[-1], row=idx + 1, col=1)

    fig.update_xaxes(title_text="Horizon step", row=n_panels, col=1)
    fig.update_layout(
        title=f"Predicted Horizon — Call {call_num}",
        height=250 * n_panels + 100,
        width=1100,
        template="plotly_white",
        showlegend=True,
    )
    return fig


def install_horizon_logger(policy, save_dir: Path):
    """Monkey-patch policy.diffusion.generate_actions to log each full horizon prediction.

    Returns a counter list [int] so the caller can read how many predictions were logged.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    call_counter = [0]
    original_generate_actions = policy.diffusion.generate_actions

    def logging_generate_actions(batch, guidance_batch):
        actions = original_generate_actions(batch, guidance_batch)
        # actions: [B, horizon, action_dim] in normalized space

        try:
            with torch.no_grad():
                action_list = torch.split(
                    actions, policy.output_sizes, dim=-1)
                unnormed = policy.unnormalize_outputs(
                    dict(zip(policy.config.action_features, action_list)))
                horizon_dict = {
                    k: v[0].cpu().numpy() for k, v in unnormed.items()
                }
            save_horizon_html(horizon_dict, call_counter[0], save_dir)
        except Exception as e:
            logger.debug(f"Horizon logging failed at call {call_counter[0]}: {e}")

        call_counter[0] += 1
        return actions

    policy.diffusion.generate_actions = logging_generate_actions
    return call_counter

