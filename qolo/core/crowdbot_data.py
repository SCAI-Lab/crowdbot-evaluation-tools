#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# =============================================================================
"""
@Author        :   Yujie He
@File          :   crowdbot_data.py
@Date created  :   2021/10/26
@Maintainer    :   Yujie He
@Email         :   yujie.he@epfl.ch
"""
# =============================================================================
"""
The module provides `CrowdBotData` and `CrowdBotDatabase` class for each
submodule to assess different subfolders and get information of containing
rosbags for each folder.
"""
# =============================================================================

import os
import yaml
import numpy as np
from pathlib import Path

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
CROWDBOT_EVAL_TOOLKIT_DIR = os.path.abspath(os.path.join(curr_dir_path, "..", ".."))
# CROWDBOT_EVAL_TOOLKIT_DIR = Path(__file__).parents[1]


def read_yaml(yaml_file):
    with open(yaml_file, encoding='utf-8') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data


class CrowdbotExpParam:
    """Class for extracting experiment parameter according to date and type"""

    def __init__(self, file, encoding="utf-8"):
        self.file = file
        self.data = read_yaml(self.file)

    def get_params(self, date, control_type):
        return self.data[date][control_type]


class CrowdBotData(object):
    """Class for extracting experiment parameter according to date and type"""

    # CROWDBOT_EVAL_TOOLKIT_DIR = Path(__file__).parents[1]
    DEFAULT_CONFIG_PATH = os.path.join(CROWDBOT_EVAL_TOOLKIT_DIR, 'data/data_path.yaml')

    def __init__(self, config=DEFAULT_CONFIG_PATH):
        self.config = config
        data_config = read_yaml(self.config)
        self.bagbase_dir = data_config['bagbase_dir']
        self.outbase_dir = data_config['outbase_dir']

    def write_yaml(self, data):
        """
        :param yaml_path
        :param data
        :param encoding
        """
        with open(self.config, 'w', encoding='utf-8') as f:
            yaml.dump(data, stream=f, allow_unicode=True)


class CrowdBotDatabase(CrowdBotData):
    def __init__(self, classdir, config=None):

        if config is None:
            super(CrowdBotDatabase, self).__init__()
        else:
            super(CrowdBotDatabase, self).__init__(config)
        data_config = read_yaml(self.config)
        self.bagbase_dir = data_config['bagbase_dir']
        self.outbase_dir = data_config['outbase_dir']

        data_processed = classdir + "_processed"
        data_processed_dir = os.path.join(self.outbase_dir, data_processed)
        if not os.path.exists(data_processed_dir):
            os.makedirs(data_processed_dir)

        # lidars/
        self.lidar_dir = os.path.join(data_processed_dir, "lidars")
        if not os.path.exists(self.lidar_dir):
            os.makedirs(self.lidar_dir)
        # lidars_nonground/
        self.lidar_nonground_dir = os.path.join(data_processed_dir, "lidars_nonground")
        if not os.path.exists(self.lidar_nonground_dir):
            os.makedirs(self.lidar_nonground_dir)
        # alg_res/[detections/tracks]
        self.alg_res_dir = os.path.join(data_processed_dir, "alg_res")
        if not os.path.exists(self.alg_res_dir):
            os.makedirs(self.alg_res_dir)
        self.dets_dir = os.path.join(self.alg_res_dir, "detections")
        if not os.path.exists(self.dets_dir):
            os.makedirs(self.dets_dir)
        self.trks_dir = os.path.join(self.alg_res_dir, "tracks")
        if not os.path.exists(self.trks_dir):
            os.makedirs(self.trks_dir)

        # source_data/[tf_qolo/pose/twist/acc/timestamp] for qolo
        self.source_data_dir = os.path.join(data_processed_dir, "source_data")
        if not os.path.exists(self.source_data_dir):
            os.makedirs(self.source_data_dir)

        # ped_data/[traj/vel] for pedestrian
        self.ped_data_dir = os.path.join(data_processed_dir, "ped_data")
        if not os.path.exists(self.ped_data_dir):
            os.makedirs(self.ped_data_dir)

        self.metrics_dir = os.path.join(data_processed_dir, "metrics")
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

        # media/[img_o3d/videos]
        self.media_dir = os.path.join(data_processed_dir, "media")
        if not os.path.exists(self.media_dir):
            os.makedirs(self.media_dir)

        # filter sequence dir from self.lidar_dir (*_stamped.npy in the same folder)
        self.seqs = [
            f
            for f in os.listdir(self.lidar_dir)
            if os.path.isdir(os.path.join(self.lidar_dir, f))
        ]
        self.seqs.sort()
        self.frames = []
        for seq in self.seqs:
            frs = os.listdir(os.path.join(self.lidar_dir, seq))
            frs.sort()
            self.frames.append(frs)

    def nr_seqs(self):
        return len(self.seqs)

    def nr_frames(self, sq_idx):
        return len(self.frames[sq_idx])

    def __getitem__(self, sq_fr_idx):
        sq_idx, fr_idx = sq_fr_idx

        assert sq_idx < self.nr_seqs(), (
            "Sequence index out of range. "
            f"Requested {sq_idx}, maximum {self.nr_seqs()}."
        )
        assert fr_idx < self.nr_frames(sq_idx), (
            "Frame index out of range. "
            f"Requested {fr_idx}, maximum {self.nr_frames(sq_idx)}."
        )

        seq = self.seqs[sq_idx]
        fr = self.frames[sq_idx][fr_idx]

        l_path = os.path.join(self.lidar_dir, seq, fr)
        lidar = np.load(l_path) if os.path.isfile(l_path) else None
        lidar = lidar.T

        l_nonground_path = os.path.join(self.lidar_nonground_dir, seq, fr)
        lidar_nonground = np.load(l_nonground_path) if os.path.isfile(l_nonground_path) else None
        lidar_nonground = lidar_nonground.T

        dnpy_all_path = os.path.join(self.dets_dir, seq + ".npy")
        tnpy_all_path = os.path.join(self.trks_dir, seq + ".npy")

        with open(dnpy_all_path, "rb") as dnpy_all:
            det_all = np.load(dnpy_all, allow_pickle=True).item()
        dets_ = det_all[fr_idx]
        dets, dets_conf = dets_[:, :-1], dets_[:, -1]

        if os.path.exists(tnpy_all_path):
            with open(tnpy_all_path, "rb") as tnpy_all:
                trk_all = np.load(tnpy_all, allow_pickle=True).item()
            trks = trk_all[fr_idx]
        else:
            trks = None

        return lidar, lidar_nonground, dets, dets_conf, trks


# filter the files with specific extensions
def bag_file_filter(f):
    if f[-4:] in [".bag"]:
        return True
    else:
        return False
