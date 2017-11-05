import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_import_adjust import data_side, data_side_aggregated
from data_import_adjust import data_bottom, data_bottom_aggregated
from data_import_adjust import data_angle, data_angle_aggregated_export

class AnglePlots:

    def angle_overtime_plot_full(self, plot_data):
        plot_out = sns.lmplot('day', 'value', palette=palette_custom_1, hue='group', col='variable', data=plot_data, x_jitter=0.5, fit_reg=False,
                   legend=False)

        sns.despine(left=True)
        plot_out.set_xlabels('Day (Post SCI)', size=15, fontweight='bold')
        plot_out.set_ylabels('Angle (degrees)', size=15, fontweight='bold')
        #plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], frameon=False, ncol=3, loc='lower center')

    def angle_overtime_plot_simple(self, plot_data):
        plot_out = sns.factorplot('day', 'value', hue='group', col='variable', palette=palette_custom_1, data=plot_data,
                                  legend=False)
        plot_out.set_axis_labels('Day (Post SCI)', 'Angle (degrees)')
        plot_out.set_titles('{col_name}')
        sns.despine(left=True)

    def angle_correlation_plot(self, plot_data, x_var, y_var, x_label, y_label):
        sns.lmplot(x_var, y_var, hue='group', data=plot_data, palette=palette_custom_1, size=5, aspect=2, legend=False)
        sns.despine()

        plt.xlabel(x_label, size=15, fontweight='bold')
        plt.ylabel(y_label, size=15, fontweight='bold')
        plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], frameon=False, loc='lower center', ncol=3)
        plt.yticks(list(np.arange(-5, 175, 25)))

#1. Defining color palette
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

#2. Subsetting relevant data
data_angle_subset = data_angle.drop(['angle_ankle'], axis=1)

#3. Adjusting angles
adjust_cols = ['angle_iliac_crest', 'angle_trochanter_major', 'angle_knee']
data_angle_subset.loc[:,adjust_cols] = data_angle_subset.loc[:,adjust_cols].applymap(lambda angle: (360-angle) if(angle>180) else angle)

#4. Melting data
data_angle_subset_melt = data_angle_subset.melt(id_vars=['RH.index', 'side', 'day', 'group'])
data_angle_aggregated_melt = data_angle_aggregated_export.melt(id_vars=['RH.index', 'day', 'group'])

#5. Creating instance of class Angleplots
angle_plot_instance = AnglePlots()

#6. Time overview simple plot
angle_plot_instance.angle_overtime_plot_simple(data_angle_aggregated_melt)

#7. Time overview full plot
angle_plot_instance.angle_overtime_plot_full(data_angle_subset_melt)

#8. Angle-to-angle correlation plots
angle_plot_instance.angle_correlation_plot(data_angle_subset, 'angle_iliac_crest', 'angle_trochanter_major', 'Angle iliac crest', 'Angle trochanter major')
angle_plot_instance.angle_correlation_plot(data_angle_subset, 'angle_trochanter_major', 'angle_knee', 'Angle trochanter major', 'Angle knee')

#9. Iliac crest height to angle correlation plots
data_angle_side = pd.concat([data_side['iliac_crest_height'], data_angle_subset], axis=1)
angle_plot_instance.angle_correlation_plot(data_angle_side, 'iliac_crest_height', 'angle_iliac_crest', 'Iliac crest height', 'Iliac crest angle')
angle_plot_instance.angle_correlation_plot(data_angle_side, 'iliac_crest_height', 'angle_knee', 'Iliac crest height', 'Knee angle')

#10. Inter knee distance to angle plots
#10A. Split datasets
angle_dictionary = dict(list(data_angle_subset.groupby(['RH.index', 'day'], as_index=False)))
bottom_dictionary = dict(list(data_bottom.groupby(['RH.index', 'day'])))
#10B. Random sample with replacement from each data subset
data_angle_reduced = pd.concat([angle_dictionary[key].sample(n=15, replace=True) for key in bottom_dictionary], ignore_index=True)
data_bottom_reduced = pd.concat([bottom_dictionary[key].sample(n=15, replace=True) for key in bottom_dictionary], ignore_index=True)
#10C. Row binding reduced/sampled datasets
data_angle_bottom_merge = pd.concat([data_angle_reduced, data_bottom_reduced.drop(['RH.index', 'day', 'group'], axis=1)], axis=1)
#10D. Creating correlation plots
angle_plot_instance.angle_correlation_plot(data_angle_bottom_merge, 'inter_knee_distance', 'angle_trochanter_major', 'Inter knee distance', 'Trochanter major angle')
angle_plot_instance.angle_correlation_plot(data_angle_bottom_merge, 'inter_knee_distance', 'angle_knee', 'Inter knee distance', 'Knee angle')