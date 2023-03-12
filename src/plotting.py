# encoding:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyplotz.pyplotz import PyplotZ
import sys

import configs
sys.dont_write_bytecode = True


class PlotGenerator:

    def __init__(self, style="whitegrid", rotation=30, palette="bright"):
        self._style = style
        self._rotation = rotation
        self._palette = palette

    @staticmethod
    def show_values(axs, orient="v", space=.01):
        def _single(ax):
            if orient == "v":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                    value = '{:.3f}'.format(p.get_height()) if p.get_height() != 0 else "0"
                    ax.text(_x, _y, value, ha="center", size="medium")
            elif orient == "h":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                    value = '{:.3f}'.format(p.get_width())
                    ax.text(_x, _y, value, ha="left", size="medium")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _single(ax)
        else:
            _single(axs)

    def count_plot(self, x, data, figwidth, figheight, new_xlabel, new_ylabel, new_title, new_fig_name=None, savefig=False, show_bar_value=False):
        pltz = PyplotZ()
        pltz.enable_chinese()
        sns.set_style(self._style)
        plt.figure(figsize=(figwidth,figheight))
        ax = sns.countplot(x=x, data=data, palette=self._palette)
        if show_bar_value:
            self.show_values(ax)
        new_xticks = list(map(lambda x: x[:5] + "\n" + x[5:] if len(x) > 5 else x, [item.get_text() for item in ax.get_xticklabels()]))
        pltz.xticks(range(len(new_xticks)), new_xticks, rotation=0, fontsize=45)
        pltz.yticks(fontsize=45)
        pltz.xlabel(new_xlabel, fontsize=50)
        pltz.ylabel(new_ylabel, fontsize=50)
        pltz.title(new_title, fontsize=60)
        if savefig:
            plt.savefig(new_fig_name, dpi=100)

    def scatter_plot(self, data, new_xlabel, new_ylabel, new_title, x=None, y=None, isgrouped=False, hue=None, new_legend_labels=None, new_legend_title=None,
                     new_fig_name=None, savefig=False, show_point_value=False):
        sns.set_style(self._style)
        pltz = PyplotZ()
        pltz.enable_chinese()
        ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=self._palette)
        if isgrouped:
            legend_labels_obj, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_labels_obj, new_legend_labels, title=new_legend_title)
        if show_point_value:
            self.show_values(ax)
        plt.xticks(rotation=self._rotation, fontsize=5)
        pltz.xlabel(new_xlabel)
        pltz.ylabel(new_ylabel)
        pltz.title(new_title)
        if savefig:
            plt.savefig(new_fig_name, dpi=100)

    def hist_plot(self, x, new_xlabel, new_ylabel, new_title, figwidth, figheight, y=None, binwidth=None, isgrouped=False, hue=None, new_legend_labels=None, new_legend_title=None,
                  new_fig_name=None, savefig=False, show_point_value=False):
        sns.set_style(self._style)
        pltz = PyplotZ()
        pltz.enable_chinese()
        plt.figure(figsize=(figwidth, figheight))
        ax = sns.histplot(x=x, y=y, hue=hue, binwidth=binwidth, palette=self._palette)
        if isgrouped:
            legend_labels_obj, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_labels_obj, new_legend_labels, title=new_legend_title)
        if show_point_value:
            self.show_values(ax)
        pltz.xticks(rotation=0, fontsize=40)
        pltz.yticks(fontsize=40)
        pltz.xlabel(new_xlabel, fontsize=45)
        pltz.ylabel(new_ylabel, fontsize=45)
        pltz.title(new_title, fontsize=60)
        if savefig:
            plt.savefig(new_fig_name, dpi=100)
        # plt.show()

    def box_plot(self, new_xlabel, new_title, x=None, y=None, isgrouped=False, hue=None, new_legend_labels=None, new_legend_title=None,
                 new_fig_name=None, savefig=False, show_point_value=False):
        sns.set_style(self._style)
        pltz = PyplotZ()
        pltz.enable_chinese()
        ax = sns.boxplot(x=x, y=y, hue=hue, palette=self._palette)
        if isgrouped:
            legend_labels_obj, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_labels_obj, new_legend_labels, title=new_legend_title)
        if show_point_value:
            self.show_values(ax)
        plt.xticks(rotation=self._rotation)
        pltz.xlabel(new_xlabel)
        pltz.title(new_title)
        if savefig:
            plt.savefig(new_fig_name, dpi=100)

    def line_plot(self, figwidth, figheight, x, y, data, new_xlabel,
                  new_ylabel, new_title, xtick_fontsize, ytick_fontsize, ylim_low, ylim_high, xlabel_fontsize,
                  ylabel_fontsize, title_fontsize, xtick_rot=0, hue=None, new_legend_labels=None, legend_fontsize=20, new_fig_name=None, savefig=False, show_point_value=False):
        pltz = PyplotZ()
        pltz.enable_chinese()
        plt.figure(figsize=(figwidth, figheight))
        sns.set_style(self._style)
        plt.ylim(ylim_low, ylim_high)
        ax = sns.lineplot(x=x, y=y, hue=hue, data=data, marker="o", markersize=12, linewidth=6, palette=self._palette)
        if show_point_value:
            self.show_values(ax)
        if hue:
            pltz.legend(labels=new_legend_labels)
            plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)
        pltz.xticks(rotation=xtick_rot, fontsize=xtick_fontsize)
        pltz.yticks(fontsize=ytick_fontsize)
        pltz.xlabel(new_xlabel, fontsize=xlabel_fontsize)
        pltz.ylabel(new_ylabel, fontsize=ylabel_fontsize)
        pltz.title(new_title, fontsize=title_fontsize)
        if savefig:
            plt.savefig(new_fig_name, dpi=100)

    def bar_plot(self, figwidth, figheight, x, y, hue, data, new_xticks, new_legend_labels, new_xlabel,
                  new_ylabel, new_title, legend_fontsize, xtick_fontsize, ytick_fontsize, xlabel_fontsize,
                 ylabel_fontsize, title_fontsize, xtick_rot=30, new_fig_name=None, savefig=False, show_point_value=False):
        pltz = PyplotZ()
        pltz.enable_chinese()
        plt.figure(figsize=(figwidth, figheight))
        sns.set_style(self._style)
        ax = sns.barplot(x=x, y=y,  hue=hue, data=data, palette=self._palette)
        if show_point_value:
            self.show_values(ax)
        current_handles, _ = plt.gca().get_legend_handles_labels()
        pltz.legend(current_handles, new_legend_labels)
        plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)
        pltz.xticks(range(len(new_xticks)), new_xticks, rotation=xtick_rot, fontsize=xtick_fontsize)
        pltz.yticks(fontsize=ytick_fontsize)
        pltz.xlabel(new_xlabel, fontsize=xlabel_fontsize)
        pltz.ylabel(new_ylabel, fontsize=ylabel_fontsize)
        pltz.title(new_title, fontsize=title_fontsize)
        if savefig:
            plt.savefig(new_fig_name, dpi=100)

    def bar_plots(self, figwidth, figheight, x, y, hue, data, new_xticks, legend_fontsize, xtick_fontsize, ytick_fontsize, xlabel_fontsize, ylabel_fontsize, title_fontsize, xtick_rot=30, new_legend_labels=None, new_xlabel=None,
                  new_ylabel=None, new_titles=None, new_fig_name=None, savefig=False, show_point_value=False):
        pltz = PyplotZ()
        pltz.enable_chinese()
        plt.figure(figsize=(figwidth, figheight))
        sns.set_style(self._style)
        for i in range(len(data)):
            plt.subplot(len(data), 1, i+1)
            ax = sns.barplot(x=x, y=y,  hue=hue, data=data[i], palette=self._palette)
            if show_point_value:
                self.show_values(ax)
            current_handles, _ = plt.gca().get_legend_handles_labels()
            pltz.legend(current_handles, new_legend_labels, handlelength=int(legend_fontsize)/6, handleheight=int(legend_fontsize)/12)
            plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)
            pltz.xticks(range(len(new_xticks)), new_xticks, rotation=xtick_rot, fontsize=xtick_fontsize)
            pltz.yticks(fontsize=ytick_fontsize)
            pltz.xlabel(new_xlabel, fontsize=xlabel_fontsize)
            pltz.ylabel(new_ylabel, fontsize=ylabel_fontsize)
            pltz.title(new_titles[i], fontsize=title_fontsize)
        plt.subplots_adjust(hspace=0.6)
        if savefig:
            plt.savefig(new_fig_name, dpi=100)
