import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_import_adjust import data_side, data_side_aggregated
from data_import_adjust import data_bottom, data_bottom_aggregated

#Creating color palette
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]









#Correlation plot
def correlation_plot(plot_var, counter_plot_var, palette_color):
    sns.kdeplot(plot_var, counter_plot_var, cmap=palette_color, shade=True, shade_lowest=False)
    sns.despine(bottom=True, left=True)

    plt.xticks(list(np.arange(2.2,2.8,0.05)))
    plt.yticks(list(np.arange(1.5,3.6,0.1)))
    plt.xlabel('Iliac crest height', size=30)
    plt.ylabel('Inter knee distance', size=30)
    plt.title('Covariation: Crest height & knee distance', size=30)

groups = list(data_side_aggregated['group'].unique())
colors = ['summer', 'autumn', 'winter']

for index, group in enumerate(groups):
    plot_var1 = data_side_aggregated[data_side_aggregated['group']==group]['iliac_crest_height_mean']
    plot_var2 = data_bottom_aggregated[data_bottom_aggregated['group']==group]['inter_knee_distance_mean']
    correlation_plot(plot_var1, plot_var2, colors[index])

#plt.savefig('corr_plot.jpg', dpi=1000)

#Time overview full
def time_overview_full(plot_data, y_var, y_label):
    sns.lmplot('day', y_var, hue='group', data=plot_data,
                          palette=palette_custom_1, x_jitter=1, size=6, aspect=2, legend=False)

    sns.despine(left=True, bottom=True)

    plt.title(y_label, size=30)
    plt.xlabel('Day (Post SCI)', size=25)
    plt.ylabel(y_label, size=25)
    plt.xticks(list(np.arange(0, 71, 7)), size=12)
    plt.yticks(list(np.arange(0, 5.5, 0.5)), size=12)
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], loc='lower center', ncol=3, fontsize=12, frameon=False)

time_overview_full(data_bottom, 'inter_knee_distance', 'Inter knee distance')
time_overview_full(data_side, 'iliac_crest_height', 'Iliac crest height')

#Time overview simple
def time_overview_simple(plot_data, x_var, y_var, group_var, y_label):
    sns.factorplot(x_var, y_var, data=plot_data, hue=group_var, aspect=3, palette=palette_custom_1, legend=False)

    plt.title(y_label, size=20)
    plt.xlabel('Day (Post SCI)', size=20)
    plt.ylabel(y_label, size=20)
    plt.xticks(list(np.arange(0, 10, 1)))
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], loc='lower center', ncol=3, fontsize=8, frameon=False)

time_overview_simple(data_side, 'day', 'iliac_crest_height', 'group', 'Iliac crest height')

#Distribution plot

distr_palette = sns.cubehelix_palette(3, rot = -0.25, light=1)
data_side['plot_group'] = data_side['day'].astype('str')+data_side['group']

plot = sns.FacetGrid(data_side, row='day', hue='group', aspect=20, size=1.5, palette=distr_palette)
plot.map(sns.kdeplot, 'iliac_crest_height', clip_on=False, shade=True, alpha=0.8, lw=1.5, bw=0.2)
plot.map(sns.kdeplot,'iliac_crest_height', clip_on=False, color='w', lw=2, bw=0.2)

for row, day in enumerate(['3', '7', '14', '21', '28']):
    plot.axes[row,0].set_ylabel('Day '+day, size=15, fontweight='bold')
plot.set_titles('')
plot.set(yticks=[])
plot.despine(left=True)
plt.xlabel('Iliac crest height', size=20, fontweight='bold')
plt.xticks(list(np.arange(1.25, 4, 0.25)))
plt.xlim([1.25,4])
