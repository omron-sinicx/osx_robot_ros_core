
#!/usr/bin/env python3

import os
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from ur_control import transformations
from osx_ur5e.real_robot.fdcc_env import FDCCEnv
from osx_ur5e.visualisation import plot_unified
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import rospy
import hydra
import signal
import sys
import numpy as np
from matplotlib import pyplot as plt
import dm_env

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="real_robot_fdcc")
def main(cfg: DictConfig):
    """
    Main function to run the grinding environment with Hydra configuration.

    Parameters:
    -----------
    cfg : DictConfig
        Hydra configuration object
    """
    rospy.init_node('osx_powder_grinding_real_robot')

    # Configure environment
    hydra_cfg = HydraConfig.get()
    if cfg.output_dir is None:
        cfg.output_dir = hydra_cfg.runtime.output_dir
    cfg.output_dir = Path(cfg.output_dir)
    os.makedirs(cfg.output_dir / cfg.tb_log_dir, exist_ok=True)
    # Print current configuration for debugging
    # print(OmegaConf.to_yaml(cfg))

    # Set the random seed for numpy
    np.random.seed(cfg.seed)

    env = FDCCEnv(config=cfg.env)

    env.go_home()

    if cfg.env.test_mode:
        print(f"{env.config.task.trajectory.initial_position=}")
        env.config.task.trajectory.initial_position[2] += 0.1

    env.get_reference_trajectory()

    max_episodes = 1
    episode_reward = 0
    traj_data = []
    force_data = []
    ref_force_data = []
    ref_traj_data = []

    for episode in tqdm(range(max_episodes), desc="Episodes"):
        env.reset(move_robot=True)
        input("Press Enter to continue...")
        env.activate_compliance_control()
        done = False

        n_steps = len(env.reference_trajectory)

        for i in tqdm(range(n_steps), desc="Steps", leave=False):
            if done:
                tqdm.write(f"Episode {episode+1} finished after {i} steps")
                tqdm.write(f"Episode reward: {episode_reward:.2f}")
                tqdm.write("------------------------")
                break

            # Generate actions based on the selected strategy
            # actions = np.zeros(env.action_dim)
            if cfg.env.controller.type == "fdcc":
                actions = {
                    "action.stiffness_diag": np.ones(6),
                }

            ts = env.step(actions)  # take action in the environment
            obs = ts.observation
            done = ts.step_type == dm_env.StepType.LAST
            episode_reward += ts.reward

            traj_data.append(np.concatenate([obs['eef_pos.position'], obs['eef_pos.quaternion']]))
            ref_traj_data.append(env.reference_trajectory[env.current_waypoint_index])
            force_data.append(obs['eef_pos.wrench'][:3])
            ref_force_data.append(env.reference_force[env.current_waypoint_index][:3])

    env.deactivate_compliance_control()
    ref_traj_data = np.array(ref_traj_data)
    traj_data = np.array(traj_data)
    print(f"{ref_traj_data.shape=}")
    print(f"{traj_data.shape=}")

    angular_error = np.array([transformations.orientation_error_as_rotation_vector(rq, aq) for rq, aq in zip(ref_traj_data[:, 3:], traj_data[:, 3:])])
    # angular_error = np.rad2deg(angular_error)
    # print(f"{angular_error.shape=}")

    # Generate plots
    if cfg.plot:
        # Create plot configuration using the new data_sources format
        plot_config = {
            "data_sources": [

                {
                    "name": "Position",
                    "reference_data": ref_traj_data,
                    "actual_data": np.array(traj_data),
                    "colors": ["blue"],
                    "plots": [
                        {
                            "name": "Z Height",
                            "mode": "1D",
                            "axes": [2],  # Only Z-axis (height)
                            "smooth_factor": 0,
                            "show_error": True,  # Show reference vs actual
                            "limits": {
                                "global": None,
                                "per_axis": {}
                            }
                        },
                        {
                            "name": "XY Trajectory",
                            "mode": "2D",
                            "axes": [0, 1],  # X, Y position for 2D trajectory
                            "smooth_factor": 0.0,  # No smoothing for 2D plots
                            "limits": {
                                "global": None,
                                "per_axis": {}
                                # Add specific limits if needed:
                                # "per_axis": {1: [env.mortar_radius-0.01, env.mortar_radius+0.01]}
                            }
                        },
                        {
                            "name": "3D Trajectory",
                            "mode": "3D",
                            "axes": [0, 1, 2],
                            "smooth_factor": 0.0,
                            "limits": {
                                "global": None,
                                "per_axis": {}
                            }
                        }
                    ]
                },
                {
                    "name": "Orientation Tracking",
                    "reference_data": np.zeros((len(angular_error), 3)),
                    "actual_data": angular_error,
                    "plots": [
                        {
                            "name": "",
                            "mode": "1D",
                            "axes": [0, 1],
                            "smooth_factor": 0.0,
                            "show_error": True,
                            "limits": {"global": None, "per_axis": {}}
                        }
                    ]
                },
                {
                    "name": "Force",
                    "reference_data": np.array(ref_force_data),
                    "actual_data": np.array(force_data),
                    "colors": ["blue"],
                    "plots": [
                        {
                            "name": "Z Error",
                            "mode": "1D",
                            "axes": [2],  # Z force component error
                            "smooth_factor": 0.7,
                            "show_error": False,  # Show error signal only
                            "limits": {
                                "global": None,
                                "per_axis": {}
                            }
                        }
                    ]
                },
            ],
            "figsize": (15, 12),  # Slightly larger to accommodate more plots
            "show_grid": True,
            "legend_location": "upper right"
        }
        figure = plot_unified(plot_config)

        # Save plot since we're using non-interactive backend
        output_file = f'{cfg.output_dir}/powder_grinding_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
        plt.show()
        plt.close('all')  # Clean up memory


if __name__ == "__main__":
    main()
