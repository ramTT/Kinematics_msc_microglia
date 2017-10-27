import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_import_adjust import data_side, data_side_aggregated
from data_import_adjust import data_bottom, data_bottom_aggregated

#1. Plot of iliac crest height over time
class DistancePlots():

    def __init__(self, data_not_aggregated, data_aggregated):
        self.data_not_aggregated = data_not_aggregated
        self.data_aggregated = data_aggregated

    def time_overview_simple(self, plot_data):
        plot_out = sns.factorplot('day', 'iliac_crest_height_mean', hue='group', data = plot_data,
                       palette=sns.color_palette('husl',8), size=5, aspect=4, legend_out=True)

        plot_out.set_axis_labels('Day (Post SCI)', 'Iliac crest height')
        plot_out.set(title='Iliac crest height')
        plot_out.add_legend(title='Study groups')

        return plot_out

    def time_overview_full(self, plot_data):
        plot_out = sns.lmplot('day', 'iliac_crest_height', hue='group', data = plot_data,
                              palette=sns.color_palette('husl',8), size=5, aspect =4, legend_out=True, x_jitter=1)

        sns.despine()
        return plot_out

    def correlation_plot(self, plot_var, counter_plot_var):
        plot_out = sns.jointplot(plot_var, counter_plot_var)
        plot_out.set_axis_labels('Iliac crest height', 'Inter knee distance')

        return plot_out
        #kde version instead https://seaborn.pydata.org/generated/seaborn.kdeplot.html
        #med full data
        #sensitivity analysis


    def distribution_plot(self):
        #http: // www.nxn.se / valent / high - contrast - stacked - distribution - plots
        #with & without bootstraping
        #för iliac crest & knee distance
        #färgkoda



distance_plot_instance = DistancePlots(data_side, data_side_aggregated)









