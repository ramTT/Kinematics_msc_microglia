import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#1. Importing class KinematicsDataAdjuster
from distance_plots_v2 import KinematicsDataAdjuster

#2. Importing data & creating variables
animal_key = pd.read_csv('animal_key_kinematics.csv')
data_set_angle = pd.read_csv('merge_side_view.csv')

keep_columns_angle = ['RH.index', 'side', 'day','Ang [deg].1', 'Ang [deg].2', 'Ang [deg].3', 'Ang [deg].4']
new_column_names_angle = {'Ang [deg].1':'angle_iliac_crest', 'Ang [deg].2':'angle_trochanter_major', 'Ang [deg].3':'angle_knee',
                                        'Ang [deg].4':'angle_ankle'}

#3. Creating instance, calling methods
instance_angle = KinematicsDataAdjuster(data_set_angle)
instance_angle.column_adjuster(keep_columns_angle, new_column_names_angle)
instance_angle.data_frame.iloc[:,3:] = instance_angle.data_frame.iloc[:,3:].applymap(lambda angle: (360-angle) if(angle>180) else angle)
instance_angle.key_data_adder(animal_key)
[instance_angle.measure_adjuster(col_name) for col_name in list(instance_angle.data_frame.columns[3:7])]
[instance_angle.indexator(col_name,'day3mean') for col_name in list(instance_angle.data_frame.columns[11:])]
keep_cols = ['RH.index', 'day', 'group']
keep_cols.extend(list(instance_angle.data_frame.columns[11:]))
instance_angle.column_cleanup(keep_cols)

data_aggregate_angle = instance_angle.aggregate_per_animal()
data_summary_angle = instance_angle.summary()

#4. Defining color palette for plotting
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

#5. Plotting raw data (angles) over time
def plot_raw_data_over_time(plot_data, y_variable, y_label):
    sns.stripplot(data = plot_data, x='day', y=y_variable, palette=palette_custom_1, jitter =0.3,
                  hue='group', dodge=True, alpha=0.7, size=10)
    sns.despine(left=True)
    plt.ylabel(y_label, size=15, fontweight='bold')
    plt.xlabel('Day (Post SCI)', size=15, fontweight='bold')
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], frameon=False, ncol=3, loc='lower center', fontsize=6)
    plt.yticks(list(np.arange(0, 3, 0.25)))

plot_raw_data_over_time(data_aggregate_angle, 'angle_iliac_crest_adjust', 'Iliac crest angle index')
plot_raw_data_over_time(data_aggregate_angle, 'angle_trochanter_major_adjust', 'Trochanter major angle index')
plot_raw_data_over_time(data_aggregate_angle, 'angle_knee_adjust', 'Knee angle index')
plot_raw_data_over_time(data_aggregate_angle, 'angle_ankle_adjust', 'Ankle angle index')

#6. Plotting aggregate data (angles) over time
def plot_aggregate_data_overtime(data_frame, y_variable, y_label):
    sns.lmplot(data=data_frame, x='day', y=y_variable, hue='group', palette=palette_custom_1,
                  x_jitter=1, lowess=True, legend_out=False, scatter_kws={'alpha':0.7, 's':200}, markers=['p','o','^'],
               size=5, aspect=2)
    sns.despine(left=True)
    plt.xlabel('Day (Post SCI)', size=12, fontweight='bold')
    plt.ylabel(y_label, size=12, fontweight='bold')
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], frameon=False, ncol=3, loc='lower center')
    #plt.yticks(list(np.arange(1, 3, 0.25)))
    plt.xticks(list(np.arange(0, 35, 7)))

plot_aggregate_data_overtime(data_aggregate_angle, 'angle_iliac_crest_adjust', 'Iliac crest angle index')
plot_aggregate_data_overtime(data_aggregate_angle, 'angle_trochanter_major_adjust', 'Trochanter major angle index')
plot_aggregate_data_overtime(data_aggregate_angle, 'angle_knee_adjust', 'Knee angle index')
plot_aggregate_data_overtime(data_aggregate_angle, 'angle_ankle_adjust', 'Ankle angle index')

#7. Plotting correlations
#A. Adding iliac crest height to angle data
from distance_plots_v2 import instance_iliac
corr_plot_data = pd.concat([instance_angle.data_frame, instance_iliac.data_frame['iliac_crest_height_adjust']], axis=1)

def correlation_plot(data_set, x_var, y_var, plot_color):
    sns.jointplot(data=data_set, x=x_var, y=y_var, kind='reg', color=plot_color, size=10)
    sns.despine(left=True, bottom=True)
    sns.set_style('white')
    plt.xlabel(' '.join(x_var.split('_')), size=15, fontweight='bold')
    plt.ylabel(' '.join(y_var.split('_')), size=15, fontweight='bold')

correlation_plot(corr_plot_data, 'angle_iliac_crest_adjust', 'iliac_crest_height_adjust', palette_custom_1[0])
correlation_plot(corr_plot_data, 'angle_trochanter_major_adjust', 'iliac_crest_height_adjust', palette_custom_1[1])
correlation_plot(corr_plot_data, 'angle_knee_adjust', 'iliac_crest_height_adjust', palette_custom_1[2])
correlation_plot(corr_plot_data, 'angle_ankle_adjust', 'iliac_crest_height_adjust', palette_custom_1[2])
#split per group