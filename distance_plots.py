import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_import_adjust import data_side, data_side_aggregated
from data_import_adjust import data_bottom, data_bottom_aggregated

#Creating color palette
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

#Correlation plot
def correlation_plot(plot_var, counter_plot_var):
    sns.kdeplot(plot_var, counter_plot_var, cmap='Blues', shade=True, shade_lowest=False)
    sns.despine(bottom=True, left=True)

    plt.xticks(list(np.arange(2.2,2.8,0.05)))
    plt.yticks(list(np.arange(1.5,3.6,0.1)))
    plt.xlabel('Iliac crest height', size=30)
    plt.ylabel('Inter knee distance', size=30)
    plt.title('Covariation: Crest height & knee distance')

correlation_plot(data_side_aggregated['iliac_crest_height_mean'], data_bottom_aggregated['inter_knee_distance_mean'])


#Time overview full
def time_overview_full(plot_data, y_var):
    sns.lmplot('day', y_var, hue='group', data=plot_data,
                          palette=palette_custom_1, x_jitter=1, size=6, aspect=2, legend=False)

    sns.despine(left=True, bottom=True)

    plt.title('Iliac crest height', size=30)
    plt.xlabel('Day (Post SCI)', size=25)
    plt.ylabel('Iliac crest height', size=25)
    plt.xticks(list(np.arange(0, 71, 7)), size=12)
    plt.yticks(list(np.arange(0, 4, 0.5)), size=12)
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], loc='lower center', ncol=3, fontsize=12)

time_overview_full(data_bottom, 'inter_knee_distance')


#Time overview simple
#def time_overview_simple(plot_data, x_var, y_var, group_var, y_label):
#    sns.tsplot(time=x_var, value=y_var, data = plot_data)

    #plt.xlabel('Day (Post SCI)', size=20)
    #plt.ylabel(y_label, size=20)
    #plt.xticks(list(np.arange(0,28,1)))
    #plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], loc='lower center', ncol=3, fontsize=8)

#time_overview_simple(data_side, 'day', 'iliac_crest_height', 'group', 'Iliac crest height')



#Distribution plots
xx = np.linspace(0, 4*np.pi, 512)
df = pd.DataFrame()

#http://www.nxn.se/valent/high-contrast-stacked-distribution-plots

for i, ind in enumerate(df):
    offset = 1.1 * i
    plt.fill_between(xx, df[ind]+offset, 0*df[ind]+offset,
                     zorder=-i,
                     facecolor='k',
                     edgecolor='w',
                     lw=3)

plt.figure()
plt.fill_between(xx, 0, 1.1)
