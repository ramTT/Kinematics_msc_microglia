import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_import_adjust import data_side, data_side_aggregated
from data_import_adjust import data_bottom, data_bottom_aggregated
from data_import_adjust import data_angle, data_angle_aggregated_export

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

#Time overview full plot
def angle_overtime_plot_full(plot_data):

    for color_num, group_name in enumerate(['sci', 'sci_medium', 'sci_msc']):
        sns.regplot('day', 'value', color=palette_custom_1[color_num], data=plot_data[plot_data['group']==group_name], x_jitter=0.5, fit_reg=False)

    sns.despine(left=True)
    plt.xlabel('Day (Post SCI)', size=15, fontweight='bold')
    plt.ylabel('Angle (degrees)', size=15, fontweight='bold')
    plt.xticks(list(np.arange(0,35,7)))

data_angle_aggregated_melt = data_angle_aggregated_export.melt(id_vars=['RH.index', 'day', 'group'])

#Time overview simple plot
def angle_overtime_plot_simple(plot_data):
    plot_out = sns.factorplot('day', 'value', hue = 'group', col = 'variable', palette = palette_custom_1, data = plot_data, legend = False)
    plot_out.set_axis_labels('Day (Post SCI)', 'Angle (degrees)')

    plot_out.set_titles('{col_name}')
    sns.despine(left=True)

angle_overtime_plot_simple(data_angle_aggregated_melt)

#Angle correlation plots
def angle_correlation_plot(plot_data, x_var, y_var, x_label, y_label):
    sns.lmplot(x_var, y_var, hue='group', data = plot_data, palette=palette_custom_1, size=5, aspect=2, legend=False)
    sns.despine()

    plt.xlabel(x_label, size=15, fontweight='bold')
    plt.ylabel(y_label, size=15, fontweight='bold')
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], frameon=False, loc='lower center', ncol=3)
    plt.yticks(list(np.arange(-5,175,25)))

angle_correlation_plot(data_angle_subset, 'angle_iliac_crest', 'angle_trochanter_major', 'Angle iliac crest', 'Angle trochanter major')
angle_correlation_plot(data_angle_subset, 'angle_trochanter_major', 'angle_knee', 'Angle trochanter major', 'Angle knee')

#Corr: iliac crest height & angles
data_angle_side = pd.concat([data_side['iliac_crest_height'], data_angle_subset], axis=1)
angle_correlation_plot(data_angle_side, 'iliac_crest_height', 'angle_iliac_crest', 'Iliac crest height', 'Iliac crest angle')
angle_correlation_plot(data_angle_side, 'iliac_crest_height', 'angle_knee', 'Iliac crest height', 'Knee angle')

#Corr: inter knee distance & angles
#Creating dataset with random samples
sample_dictionary = dict(list(data_side.groupby(['RH.index', 'day'], as_index=False)))
pd.concat([sample_dictionary[key].sample(frac=0.2) for key in sample_dictionary], ignore_index=True)













#Creating time overview plots for angle
for angle_type in list(data_angle_subset_melt.variable.unique()):
    plt.figure()
    angle_overtime_plot_full(data_angle_subset_melt[data_angle_subset_melt['variable']==angle_type])