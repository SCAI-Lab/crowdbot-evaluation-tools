# -*-coding:utf-8 -*-
"""
@File    :   eval_util.py
@Time    :   2021/11/16
@Author  :   Yujie He
@Version :   1.0
@Contact :   yujie.he@epfl.ch
@State   :   Dev
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def save_motion_img(qolo_command_dict, qolo_eval_dict, base_dir, seq_name, suffix):
    ts = qolo_command_dict["timestamp"]
    start_ts = qolo_eval_dict.get("start_command_ts")
    duration2goal = qolo_eval_dict.get("duration2goal")

    new_start_ts = np.max([start_ts - np.min(ts), 0.0])
    new_end_ts = new_start_ts + duration2goal

    plot_attr = ("x_vel", "zrot_vel", "x_acc", "zrot_acc", "x_jerk", "zrot_jerk")
    unit = (
        "$V$ [$m/s$]",
        "$V_w$ [$rad/s$]",
        "$a$ [$m/s^2$]",
        "$a_w$ [$rad/s^2$]",
        "$J$ [$m/s^3$]",
        "$J_w$ [$rad/s^3$]",
    )

    # ref: https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
    fig, ax = plt.subplots(3, 2, sharex="col", figsize=(10, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html#sphx-glr-gallery-subplots-axes-and-figures-axes-zoom-effect-py
    # ref: https://matplotlib.org/stable/gallery/shapes_and_collections/artist_reference.html#sphx-glr-gallery-shapes-and-collections-artist-reference-py

    for i in range(3):
        for j in range(2):
            xx = ts - np.min(ts)
            yy = qolo_command_dict[plot_attr[i * 2 + j]]
            ax[i, j].plot(xx, yy, linewidth=0.8, color="purple")
            ax[i, j].axvline(x=new_start_ts, linestyle="--", linewidth=1.5, color="red")
            ax[i, j].axvline(x=new_end_ts, linestyle="--", linewidth=1.5, color="red")

            # ref: https://stackoverflow.com/questions/50753721/can-not-reset-the-axes
            # print(type(np.max(yy)))
            # print(type(np.abs(np.min(yy))))
            # TypeError: 'numpy.float64' object cannot be interpreted as an integer
            # y_lim = np.max(np.max(yy), np.abs(np.min(yy)))
            if np.max(yy) >= np.abs(np.min(yy)):
                y_lim = np.max(yy)
            else:
                y_lim = np.abs(np.min(yy))
            rect_up = mpatches.Rectangle(
                (new_start_ts, 0),
                duration2goal,
                100.0 * y_lim,
                facecolor="red",
                alpha=0.2,
            )  # xy, width, height
            ax[i, j].add_patch(rect_up)
            rect_down = mpatches.Rectangle(
                (new_start_ts, -100.0 * y_lim),
                duration2goal,
                100.0 * y_lim,
                facecolor="red",
                alpha=0.2,
            )
            ax[i, j].add_patch(rect_down)
            # TODO: cannot show correctly when `MAX=nan, MIN=-nan` occurs
            # print("MAX={}, MIN=-{}".format(max_y, min_y))

            ax[i, j].set_ylabel(unit[i * 2 + j])
            if i == 2:
                ax[i, j].set_xlabel("t [s]")
            if i == 0:  # only nominal velocity is always nonnegative
                if i == 0 and j == 0:
                    ax[i, j].set_ylim(bottom=-1.5, top=1.5)
                elif i == 0 and j == 1:
                    ax[i, j].set_ylim(bottom=-5, top=5)

    fig.tight_layout()
    qolo_img_path = os.path.join(
        base_dir, seq_name + suffix + ".png"
    )  # "_qolo_command"
    plt.savefig(qolo_img_path, dpi=300)  # png, pdf
