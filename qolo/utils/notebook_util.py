#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# =============================================================================
"""
@Author        :   Yujie He
@File          :   notebook_util.py
@Date created  :   2021/11/24
@Maintainer    :   Yujie He
@Email         :   yujie.he@epfl.ch
"""
# =============================================================================
"""
The module provides utility functions used in notebook for categorical plot and customized folder traverse functions.
"""
# =============================================================================

import os
import numpy as np
import os.path as path
import pandas as pd

import matplotlib
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

plt.ioff()
import seaborn as sns

from qolo.core.crowdbot_data import CrowdBotDatabase


def values2color_list(
    value_list, cmap_name='hot', range=(0.1, 0.9), reverse=True, given_values=None
):
    value_unique = np.unique(value_list)
    value_len = len(value_unique)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    if given_values is not None:
        value_normalized = given_values
    else:
        value_normalized = np.linspace(range[0], range[1], num=value_len)
    if reverse:
        value_normalized = np.flip(value_normalized)
    color_unique = []
    for value in value_unique:
        index = np.where(value_unique == value)[0][0]
        color_unique.append(cmap(value_normalized[index]))

    return value_unique, color_unique


def values2colors(value_list, cmap_name='hot', range=(0.1, 0.9), reverse=True):
    # ref: https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib

    value_unique = np.unique(value_list)
    value_len = len(value_unique)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    value_normalized = np.linspace(range[0], range[1], num=value_len)
    if reverse:
        value_normalized = np.flip(value_normalized)
    color_list = []
    for value in value_list:
        index = np.where(value_unique == value)[0][0]
        color_list.append(cmap(value_normalized[index]))
    return color_list


def set_box_color(bp, color):
    """
    Input:
        bp: boxplot instance
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def adjust_box_widths(g, fac):
    """Adjust the withs of a seaborn-generated boxplot."""
    ##iterating through Axes instances
    for ax in g.axes.flatten():

        ##iterating through axes artists:
        for c in ax.get_children():

            ##searching for PathPatches
            if isinstance(c, PathPatch):
                ##getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                ##setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                ##setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


# derived from https://stackoverflow.com/a/53380401/7961693
def walk(top, topdown=True, onerror=None, followlinks=False, maxdepth=None):
    islink, join, isdir = path.islink, path.join, path.isdir
    names = os.listdir(top)
    dirs, nondirs = [], []
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    if topdown:
        yield top, dirs, nondirs

    if maxdepth is None or maxdepth > 1:
        for name in dirs:
            new_path = join(top, name)
            if followlinks or not islink(new_path):
                for x in walk(
                    new_path,
                    topdown,
                    onerror,
                    followlinks,
                    None if maxdepth is None else maxdepth - 1,
                ):
                    yield x
    if not topdown:
        yield top, dirs, nondirs


def violinplot(axes, df, metric, category, title, ylabel, ylim):

    sns.violinplot(x=category, y=metric, data=df, ax=axes)

    axes.yaxis.grid(True)

    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.set_ylim(bottom=ylim[0], top=ylim[1])

    plt.show()


def categorical_plot(
    axes,
    df,
    metric,
    category,
    xlabel,
    ylabel,
    ylim,
    title=None,
    group=None,
    lgd_labels=None,
    lgd_fontsz=12,
    lgd_font=None,
    loc='lower right',
    kind='violin',
    titlefontsz=16,
    yint=False,
    scatter_palette="colorblind",
    box_palette="hot",  # Blues
):

    sns.set_theme(style="whitegrid")

    # fmt: off
    # use stripplot (less points) instead of swarmplot to handle many datapoints
    sns.swarmplot(x=category, y=metric, hue=group, data=df, ax=axes,
                  size=6, alpha=0.8, palette=scatter_palette,
                  edgecolor='black', dodge=True,
                 )
    if kind == 'violin':
        sns.violinplot(x=category, y=metric, hue=group, data=df, ax=axes,
                       linewidth=1.1, notch=False, orient="v",
                       dodge=True, palette=box_palette, inner=None,
                      )
    elif kind == 'box':
        sns.boxplot(x=category, y=metric, hue=group, data=df, ax=axes,
                    linewidth=2, notch=False, orient="v",
                    dodge=True, palette=box_palette,
                   )

    # sns.despine(trim=True)

    # fmt: on
    axes.yaxis.grid(True)

    # legend font
    lgd_font = font_manager.FontProperties(
        family=lgd_font, style='normal', size=lgd_fontsz
    )

    if group:
        # deduplicate labels
        # method1: https://stackoverflow.com/a/33440601/7961693
        # hand, labl = ax.get_legend_handles_labels()
        # plt.legend(np.unique(labl))

        # method2: https://stackoverflow.com/a/33424628/7961693
        lablout, handout = [], []
        hand, labl = axes.get_legend_handles_labels()
        for h, l in zip(hand, labl):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)
        if lgd_labels:
            axes.legend(handles=handout, labels=lgd_labels, loc=loc, prop=lgd_font)
        else:
            axes.legend(handles=handout, labels=lablout, loc=loc, prop=lgd_font)

    if title is not None:
        axes.set_title(title, fontweight='bold', fontsize=titlefontsz)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_ylim(bottom=ylim[0], top=ylim[1])
    if yint:
        print("Force to int ylabel!")
        # https://stackoverflow.com/a/11417609
        ya = axes.get_yaxis()
        from matplotlib.ticker import MaxNLocator

        ya.set_major_locator(MaxNLocator(integer=True))


def barplot_annotate_brackets(
    num1,
    num2,
    data,
    center,
    height,
    line_y=None,
    yerr=None,
    dh=0.05,
    barh=0.05,
    fs=None,
    maxasterix=None,
):
    """
    Annotate barplot with p-values.
    Original: https://stackoverflow.com/a/52333561/13954301
    Modified by Yujie He

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    if type(data) is str:
        text = data
    else:
        """
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        p = .05
        text = ''


        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'
        """
        # p<0.1 --> *
        # p<0.05 --> **
        # p<0.01 --> ***
        if 0.05 < data <= 0.1:
            text = '*'
        elif 0.01 < data <= 0.05:
            text = '**'
        elif data <= 0.01:
            text = '***'
        else:
            text = ''
            # text = 'n. s.'

    if len(text) > 0:

        lx, ly = center[num1], height[num1]
        rx, ry = center[num2], height[num2]

        if yerr:
            ly += yerr[num1]
            ry += yerr[num2]

        ax_y0, ax_y1 = plt.gca().get_ylim()
        dh *= ax_y1 - ax_y0
        barh *= ax_y1 - ax_y0

        if line_y is not None:
            barx = [lx, lx, rx, rx]
            bary = [line_y, line_y + barh, line_y + barh, line_y]
            mid = ((lx + rx) / 2, line_y + barh)
        else:
            y = max(ly, ry) + dh

            barx = [lx, lx, rx, rx]
            bary = [y, y + barh, y + barh, y]
            mid = ((lx + rx) / 2, y + barh)

        plt.plot(barx, bary, c='black')

        kwargs = dict(ha='center', va='bottom')
        if fs is not None:
            kwargs['fontsize'] = fs

        plt.text(*mid, text, **kwargs)


def main():
    heights = [1.8, 2, 3]
    bars = np.arange(len(heights))

    plt.figure()
    plt.bar(bars, heights, align='center')
    plt.ylim(0, 5)
    barplot_annotate_brackets(0, 1, 0.1, bars, heights)
    barplot_annotate_brackets(1, 2, 0.001, bars, heights)
    barplot_annotate_brackets(0, 2, 'p < 0.0075', bars, heights, dh=0.2)
    plt.show()


def import_eval_res(
    eval_dirs,
    crowd_metrics=None,
    path_metrics=None,
    control_metrics=None,
    travel_path_thres=5.0,
    config=None,
):
    """Load generated results as pd.DataFrame"""
    if not crowd_metrics:
        crowd_metrics = (
            'avg_crowd_density2_5',
            'std_crowd_density2_5',
            'max_crowd_density2_5',
            'avg_crowd_density5',
            'std_crowd_density5',
            'max_crowd_density5',
            'avg_min_dist',
            'virtual_collision',
        )

    if not path_metrics:
        path_metrics = (
            'rel_duration2goal',
            'rel_path_length2goal',
            'path_length2goal',
            'duration2goal',
            'min_dist2goal',
        )

    if not control_metrics:
        control_metrics = (
            'rel_jerk',
            'avg_fluency',
            'contribution',
            'avg_agreement',
        )

    frames = []

    for eval_dir in eval_dirs:

        # extract date
        date = eval_dir[:4]
        control_type = eval_dir[5:]

        print("Reading results from {}".format(eval_dir))

        # new a CrowdBotDatabase() instance
        eval_database = CrowdBotDatabase(classdir=eval_dir, config=config)

        eval_dict = {'seq': eval_database.seqs}
        eval_dict.update(
            {'control_type': [control_type for i in range(eval_database.nr_seqs())]}
        )

        eval_dict.update({k: [] for k in crowd_metrics})
        eval_dict.update({k: [] for k in path_metrics})
        eval_dict.update({k: [] for k in control_metrics})

        for idx, seq in enumerate(eval_database.seqs):
            eval_res_dir = os.path.join(eval_database.metrics_dir)

            crowd_eval_npy = os.path.join(eval_res_dir, seq, seq + "_crowd_eval.npy")
            crowd_eval_dict = np.load(
                crowd_eval_npy,
                allow_pickle=True,
            ).item()
            for iidx, val in enumerate(crowd_metrics):
                eval_dict[crowd_metrics[iidx]].append(crowd_eval_dict[val])

            path_eval_npy = os.path.join(eval_res_dir, seq, seq + "_path_eval.npy")
            path_eval_dict = np.load(
                path_eval_npy,
                allow_pickle=True,
            ).item()
            for iidx, val in enumerate(path_metrics):
                eval_dict[path_metrics[iidx]].append(path_eval_dict[val])

            qolo_eval_npy = os.path.join(eval_res_dir, seq, seq + "_qolo_eval.npy")
            qolo_eval_dict = np.load(
                qolo_eval_npy,
                allow_pickle=True,
            ).item()
            for iidx, val in enumerate(control_metrics):
                eval_dict[control_metrics[iidx]].append(qolo_eval_dict[val])

        eval_df = pd.DataFrame(eval_dict)
        eval_df.columns = (
            ['seq', 'control_type']
            + list(crowd_metrics)
            + list(path_metrics)
            + list(control_metrics)
        )

        # Filter path_length2goal less than 5 meter
        eval_df = eval_df[eval_df.path_length2goal >= travel_path_thres]
        # add date col
        eval_df['date'] = [date] * len(eval_df)

        frames.append(eval_df)

    return pd.concat(frames, ignore_index=True)
