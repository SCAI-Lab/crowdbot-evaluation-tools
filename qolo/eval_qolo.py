#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# =============================================================================
"""
@Author        :   Yujie He
@File          :   eval_qolo.py
@Date created  :   2021/11/09
@Maintainer    :   Yujie He
@Email         :   yujie.he@epfl.ch
"""
# =============================================================================
"""
The module provides the evaluation pipeline includeing the computation of
path efficiency and share control-related metrics (relative time to goal,
relative time to goal, relative jerk, agreement, fluency and corresponding
visualization.
The emulation results is exported with suffix as "_qolo_eval.npy".
"""
# =============================================================================
"""
TODO:
1. check data source from pose2d (odom) or tf_qolo
"""
# =============================================================================

import os
import argparse

import numpy as np

from crowdbot_data import CrowdBotDatabase
from eval_res_plot import save_motion_img, save_path_img
from metric_qplo_perf import (
    compute_time_path,
    compute_fluency,
    compute_agreement,
    compute_relative_jerk,
)

#%% main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert data from rosbag")

    parser.add_argument(
        "-b",
        "--base",
        default="/home/crowdbot/Documents/yujie/crowdbot_tools",
        type=str,
        help="base folder, i.e., the path of the current workspace",
    )
    parser.add_argument(
        "-d",
        "--data",
        default="data",
        type=str,
        help="data folder, i.e., the name of folder that stored extracted raw data and processed data",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="0421_mds",
        type=str,
        help="different subfolder in rosbag/ dir",
    )
    parser.add_argument(
        "--save_img",
        dest="save_img",
        action="store_true",
        help="plot and save crowd density image",
    )
    parser.set_defaults(save_img=True)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Whether to overwrite existing rosbags (default: false)",
    )
    parser.set_defaults(overwrite=False)
    parser.add_argument(
        "--replot",
        dest="replot",
        action="store_true",
        help="Whether to re-plot existing images (default: false)",
    )
    parser.set_defaults(replot=False)
    parser.add_argument(
        "--goal_len", default=20.0, type=float, help="The length to travel in the test"
    )
    args = parser.parse_args()

    cb_data = CrowdBotDatabase(args.folder)

    print("Starting evaluating qolo from {} sequences!".format(cb_data.nr_seqs()))

    eval_res_dir = os.path.join(cb_data.metrics_dir)

    if not os.path.exists(eval_res_dir):
        print("Result images and npy will be saved in {}".format(eval_res_dir))
        os.makedirs(eval_res_dir, exist_ok=True)

    for seq_idx in range(cb_data.nr_seqs()):

        seq = cb_data.seqs[seq_idx]
        print(
            "({}/{}): {} with {} frames".format(
                seq_idx + 1, cb_data.nr_seqs(), seq, cb_data.nr_frames(seq_idx)
            )
        )

        # load pose2d
        pose2d_dir = os.path.join(cb_data.source_data_dir, "pose2d")
        qolo_pose2d_path = os.path.join(pose2d_dir, seq + "_pose2d.npy")
        if not os.path.exists(qolo_pose2d_path):
            print("ERROR: Please extract pose2d by using pose2d2npy.py")
        qolo_pose2d = np.load(qolo_pose2d_path, allow_pickle=True).item()

        # load twist, qolo_command
        twist_dir = os.path.join(cb_data.source_data_dir, "twist")
        qolo_twist_path = os.path.join(twist_dir, seq + "_twist_raw.npy")
        command_sampled_filepath = os.path.join(twist_dir, seq + "_qolo_command.npy")
        if (not os.path.exists(qolo_twist_path)) or (
            not os.path.exists(command_sampled_filepath)
        ):
            print("ERROR: Please extract twist_stamped by using twist2npy.py")
        qolo_twist = np.load(qolo_twist_path, allow_pickle=True).item()
        qolo_command_dict = np.load(command_sampled_filepath, allow_pickle=True).item()

        # load qolo_state
        tfqolo_dir = os.path.join(cb_data.source_data_dir, "tf_qolo")
        qolo_state_filepath = os.path.join(tfqolo_dir, seq + "_qolo_state.npy")
        qolo_lidarstamp_filepath = os.path.join(tfqolo_dir, seq + "_tfqolo_sampled.npy")
        if not os.path.exists(qolo_state_filepath):
            print("ERROR: Please extract twist_stamped by using tfqolo2npy.py")
        qolo_state_dict = np.load(qolo_state_filepath, allow_pickle=True).item()
        qolo_lidarstamp_dict = np.load(
            qolo_lidarstamp_filepath, allow_pickle=True
        ).item()

        # load commands
        cmd_dir = os.path.join(cb_data.source_data_dir, "commands")
        cmd_raw_filepath = os.path.join(cmd_dir, seq + "_commands_raw.npy")
        if not os.path.exists(cmd_raw_filepath):
            print("ERROR: Please extract twist_stamped by using commands2npy.py")
        cmd_raw_dict = np.load(cmd_raw_filepath, allow_pickle=True).item()

        # 1. compute (start_ts, end_idx, end_ts, duration2goal, path_length2goal)
        time_path_computed = compute_time_path(qolo_twist, qolo_pose2d, args.goal_len)

        # dest: seq+'_crowd_eval.npy' file in eval_res_dir
        qolo_eval_npy = os.path.join(eval_res_dir, seq + "_qolo_eval.npy")

        # only for plotting function update!
        if args.replot:
            qolo_eval_npy = np.load(qolo_eval_npy, allow_pickle=True).item()

            # figure1: path
            save_path_img(qolo_pose2d, time_path_computed, eval_res_dir, seq)

            # figure2: viz twist, acc, jerk from qolo_command and qolo_state
            save_motion_img(
                qolo_command_dict,
                qolo_eval_dict,
                eval_res_dir,
                seq,
                suffix="_qolo_command",
            )
            save_motion_img(
                qolo_state_dict,
                qolo_eval_dict,
                eval_res_dir,
                seq,
                suffix="_qolo_state",
            )

            print("Replot images!")
        else:
            if (not os.path.exists(qolo_eval_npy)) or (args.overwrite):

                # timestamp can be read from lidars/ folder
                stamp_file_path = os.path.join(cb_data.lidar_dir, seq + "_stamped.npy")
                lidar_stamped_dict = np.load(stamp_file_path, allow_pickle=True)
                ts = lidar_stamped_dict.item().get("timestamp")

                start_cmd_ts = time_path_computed[0]
                end_cmd_ts = time_path_computed[1]

                attrs = ("jerk", "agreement", "fluency")

                # 2. jerk
                qolo_eval_dict = dict()
                cmd_ts = qolo_command_dict["timestamp"]
                x_jerk = qolo_command_dict["x_jerk"]
                zrot_jerk = qolo_command_dict["zrot_jerk"]

                # relative jerk
                rel_jerk = compute_relative_jerk(
                    x_jerk, zrot_jerk, cmd_ts, start_cmd_ts, end_cmd_ts
                )
                qolo_eval_dict.update({"rel_jerk": rel_jerk})
                qolo_eval_dict.update({"avg_linear_jerk": np.average(x_jerk)})
                qolo_eval_dict.update({"avg_angular_jerk": np.average(zrot_jerk)})

                # 3. fluency
                fluency = compute_fluency(cmd_raw_dict, start_cmd_ts, end_cmd_ts)
                qolo_eval_dict.update({"avg_fluency": fluency[0]})
                qolo_eval_dict.update({"std_fluency": fluency[1]})

                # 4. agreement with command (sampled)
                control_type = args.folder[5:]
                agreement_contri = compute_agreement(
                    cmd_raw_dict, start_cmd_ts, end_cmd_ts, control_type
                )
                qolo_eval_dict.update({"contribution": agreement_contri[0]})
                qolo_eval_dict.update({"avg_agreement": agreement_contri[1]})
                qolo_eval_dict.update({"std_agreement": agreement_contri[2]})
                qolo_eval_dict.update({"avg_linear_diff": agreement_contri[5]})
                qolo_eval_dict.update({"avg_angular_diff": agreement_contri[7]})

                # 4. add path related-metrics
                qolo_eval_dict.update({"start_command_ts": start_cmd_ts})
                qolo_eval_dict.update({"end_command_ts": end_cmd_ts})
                qolo_eval_dict.update({"duration2goal": time_path_computed[2]})
                qolo_eval_dict.update({"path_lenth2goal": time_path_computed[3]})
                qolo_eval_dict.update({"goal_reached": time_path_computed[5]})
                qolo_eval_dict.update({"min_dist2goal": time_path_computed[6]})
                qolo_eval_dict.update({"rel_duration2goal": time_path_computed[8]})
                qolo_eval_dict.update({"rel_path_length2goal": time_path_computed[9]})

                np.save(qolo_eval_npy, qolo_eval_dict)

                if args.save_img:
                    # figure1: path
                    save_path_img(qolo_pose2d, time_path_computed, eval_res_dir, seq)

                    # figure2: viz twist, acc, jerk
                    save_motion_img(
                        qolo_command_dict,
                        qolo_eval_dict,
                        eval_res_dir,
                        seq,
                        suffix="_qolo_command",
                    )
                    save_motion_img(
                        qolo_state_dict,
                        qolo_eval_dict,
                        eval_res_dir,
                        seq,
                        suffix="_qolo_state",
                    )
            else:
                print(
                    "Detecting the generated {} already existed!".format(qolo_eval_npy)
                )
                print(
                    "Will not overwrite. If you want to overwrite, use flag --overwrite"
                )
                continue

    print("Finish qolo evaluation!")
