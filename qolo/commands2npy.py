# -*-coding:utf-8 -*-
"""
@File    :   commands2npy.py
@Time    :   2021/11/10
@Author  :   Yujie He
@Version :   1.0
@Contact :   yujie.he@epfl.ch
@State   :   Dev
"""

# TODO: unpack in different dict format

import os
import sys
import argparse

import numpy as np

import rosbag

from crowdbot_data import AllFrames, bag_file_filter
from process_util import interp_translation

#%% extract commands from `rds_network_ros/ToGui` in  rosbag without rosbag play
def ts_to_sec(ts):
    """convert ros timestamp into second"""
    return ts.secs + ts.nsecs / float(1e9) # "{:.12f}".format(t)


def extract_cmd_from_rosbag(bag_file_path, args):
    command_msg_sum = 0
    num_msgs_between_logs = 100
    # x_list, zrot_list, t_list = [], [], []

    with rosbag.Bag(bag_file_path, "r") as bag:
        # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)
        total_num_command_msgs = bag.get_message_count(topic_filters=args.to_gui_topic)
        print(
            "Found twist topic: {} with {} messages".format(
                args.to_gui_topic, total_num_command_msgs
            )
        )

        # Extract twist msg to file
        # !!!for topic, msg, t in bag.read_messages(topics=['/topics name'])
        for topic, msg, t in bag.read_messages():
            if topic == args.to_gui_topic:
                time = t.to_sec()  # these messages do not have time stamps
                commands = np.array(
                    [
                        msg.nominal_command.linear,
                        msg.nominal_command.angular,
                        msg.corrected_command.linear,
                        msg.corrected_command.angular,
                    ]
                )
                # x_vel    = msg.twist.linear.x
                # zrot_vel = msg.twist.angular.z
                # ts_vel   = ts_to_sec(msg.header.stamp)
                # x_list.append(x_vel)
                # zrot_list.append(zrot_vel)
                # t_list.append(ts_vel)

                if (
                    command_msg_sum % num_msgs_between_logs == 0
                    or command_msg_sum >= total_num_command_msgs - 1
                ):
                    print(
                        "twist messages: {} / {}".format(
                            command_msg_sum + 1, total_num_command_msgs
                        )
                    )
                command_msg_sum += 1

    # t_np = np.asarray(t_list, dtype=np.float64)
    # x_np = np.asarray(x_list, dtype=np.float64)
    # z_np = np.asarray(zrot_list, dtype=np.float64)

    cmd_stamped_dict = {"timestamp": t_np, "x": x_np, "zrot": z_np}
    print("Current twist extracted!")
    return cmd_stamped_dict


def interp_twist(cmd_stamped_dict, target_dict):
    """Calculate interpolations for all states"""
    source_ts = cmd_stamped_dict.get("timestamp")
    source_x = cmd_stamped_dict.get("x")
    source_zrot = cmd_stamped_dict.get("zrot")
    interp_ts = target_dict.get("timestamp")
    interp_ts = np.asarray(interp_ts, dtype=np.float64)
    # don't resize, just discard the timestamps smaller than source
    if min(interp_ts) < min(source_ts):
        interp_ts[interp_ts < min(source_ts)] = min(source_ts)
    elif max(interp_ts) > max(source_ts):
        interp_ts[interp_ts > max(source_ts)] = max(source_ts)

    interp_dict = {}
    interp_dict["timestamp"] = interp_ts
    interp_dict["x"] = interp_translation(source_ts, interp_ts, source_x)
    interp_dict["zrot"] = interp_translation(source_ts, interp_ts, source_zrot)
    return interp_dict


#%% main file
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
        default="nocam_rosbags",
        type=str,
        help="different subfolder in rosbag/ dir",
    )
    parser.add_argument(
        "--to_gui_topic",
        default="/rds_to_gui",
        type=str,
        help="topic for qolo rds_to_gui",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Whether to overwrite existing rosbags (default: false)",
    )
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()

    allf = AllFrames(args)

    # source: rosbag data in data/rosbag/xxxx
    rosbag_dir = os.path.join(args.base, args.data, "rosbag", args.folder)
    bag_files = list(filter(bag_file_filter, os.listdir(rosbag_dir)))

    # destination: twist data in data/xxxx_processed/source_data/commands
    cmd_dir = os.path.join(allf.source_data_dir, "commands")
    if not os.path.exists(cmd_dir):
        os.makedirs(cmd_dir)

    print("Starting extracting command msg from {} rosbags!".format(len(bag_files)))

    counter = 0
    for bf in bag_files:
        bag_path = os.path.join(rosbag_dir, bf)
        bag_name = bf.split(".")[0]
        counter += 1
        print("({}/{}): {}".format(counter, len(bag_files), bag_path))

        cmd_stamped_filepath = os.path.join(cmd_dir, bag_name + "_commands.npy")
        # sample with lidar frame
        cmd_sampled_filepath = os.path.join(cmd_dir, bag_name + "_commands_sampled.npy")

        if not os.path.exists(cmd_sampled_filepath) or (args.overwrite):
            if (not os.path.exists(cmd_stamped_filepath)) or (args.overwrite):
                cmd_stamped_dict = extract_cmd_from_rosbag(bag_path, args)
                np.save(cmd_stamped_filepath, cmd_stamped_dict)
            else:
                print(
                    "Detecting the generated {} already existed!".format(
                        cmd_stamped_filepath
                    )
                )
                print("If you want to overwrite, use flag --overwrite")
                cmd_stamped_dict = np.load(
                    cmd_stamped_filepath, allow_pickle=True
                ).item()

            lidar_stamped = np.load(
                os.path.join(allf.lidar_dir, bag_name + "_stamped.npy"),
                allow_pickle=True,
            ).item()
            twist_sampled_dict = interp_twist(cmd_stamped_dict, lidar_stamped)
            np.save(cmd_sampled_filepath, twist_sampled_dict)

            source_len = len(cmd_stamped_dict["timestamp"])
            target_len = len(lidar_stamped["timestamp"])
            print(
                "# Sample from {} frames to {} frames.".format(source_len, target_len)
            )

    print("Finish extracting all commands!")
