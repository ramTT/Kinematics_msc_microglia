import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#Importing datasets and adjusting
from positioning_coordination_v2 import data_frame_boostrap_raw, data_frame_boostrap_aggregate, data_frame_boostrap_summary

#Raw - iliac
df_stat_raw = data_frame_boostrap_raw.copy()
df_stat_raw = df_stat_raw.rename(columns={'x_value':'x', 'y_value':'y'})
#Aggregate - iliac
df_stat_aggregate = data_frame_boostrap_aggregate.copy()
df_stat_aggregate = df_stat_aggregate.rename(columns={'mean_x':'x', 'mean_y':'y'})
df_stat_aggregate.drop(['std_x', 'std_y'], axis=1, inplace=True)
#Summary - iliac
df_stat_summary = data_frame_boostrap_summary.copy()
df_stat_summary.drop(['std_x', 'std_y', 'std_norm'], axis=1, inplace=True)
df_stat_summary = df_stat_summary.rename(columns={'mean_x':'x', 'mean_y':'y'})

from positioning_coordination_knee import instance_knee
#Raw - knee
df_stat_raw_knee = instance_knee.data_frame.copy()
df_stat_raw_knee.drop(['displacement', 'x', 'y', 'inter_knee_distance'], axis=1, inplace=True)
#Aggregate - knee
df_stat_aggregate_knee = df_stat_raw_knee.groupby(['RH.index', 'day', 'group'], as_index=False).mean()
#Summary - knee
df_stat_summary_knee = df_stat_aggregate_knee.groupby(['day', 'group'], as_index=False).mean()
df_stat_summary_knee.drop(['RH.index'], axis=1, inplace=True)

############################################################## DEFINING COLOR PALETTES ##########################################################################
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

def group_2_color(argument):
    '''Dictionary mapping (replacing switch/case statement)'''
    switcher = {
        'sci': palette_custom_1[0],
        'sci_medium': palette_custom_1[1],
        'sci_msc': palette_custom_1[2]
    }
    return switcher.get(argument)

def joint_2_shape(argument):
    marker_list = ['+', 'o', '*', 'v', 'd']
    switcher = {
        'ankle': marker_list[0],
        'iliac':marker_list[1],
        'knee':marker_list[2],
        'toe':marker_list[3],
        'trochanter':marker_list[4]
    }
    return switcher.get(argument)

############################################################### STATISTICAL ANALYSIS ###########################################################################

#A1. Comparing joint positions between groups at each time point for each joint
def joint_comparison_between_group(data_set, plot_day, var_type):
    data_set = data_set[data_set['day']==plot_day]
    joints = list(data_set['joint_name'].unique())

    def kruskal_test(joint_name):
        data_set_dict = dict(list(data_set[data_set['joint_name']==joint_name].groupby(['group'])))
        kruskal_out = stats.kruskal(data_set_dict['sci'][var_type], data_set_dict['sci_medium'][var_type], data_set_dict['sci_msc'][var_type])
        return (plot_day, joint_name, kruskal_out[1])

    return pd.DataFrame([kruskal_test(joint) for joint in joints], columns=['day','joint', 'p_value'])

pd.concat([joint_comparison_between_group(df_stat_aggregate, day, 'x') for day in list(df_stat_aggregate['day'].unique())], axis=0, ignore_index=True)
pd.concat([joint_comparison_between_group(df_stat_aggregate, day, 'y') for day in list(df_stat_aggregate['day'].unique())], axis=0, ignore_index=True)

#Steady state analysis (iliac crest height and inter knee distance)
def steady_state_random_sampler(data_set, study_group, day_start):
    data_set = data_set[(data_set['day']>=day_start) & (data_set['group']==study_group)]
    data_set_dict = dict(list(data_set.groupby(['joint_name'])))

    def within_joint_bootstrapper(key):
        df = pd.DataFrame(data_set_dict[key])
        df = df.sample(frac=1, replace=True).agg({'x':['mean'], 'y':['mean']})
        return df
    #Calling within_joint_bootstrapper for each joint and adding joint and group name
    df_out = pd.concat([within_joint_bootstrapper(key) for key in list(data_set_dict.keys())], axis=0, ignore_index=True)
    df_out['joint_name'] = list(data_set_dict.keys())
    df_out['group'] = study_group
    return df_out

def steady_state_random_sampler_caller(data_set, n_runs, day_min):
    study_groups = ['sci', 'sci_medium', 'sci_msc']
    def group_looper():
        return pd.concat([steady_state_random_sampler(data_set, group, day_min) for group in study_groups], axis=0, ignore_index=True)

    return pd.concat([group_looper() for _ in range(n_runs)], axis=0, ignore_index=True)

df_bootstrap = steady_state_random_sampler_caller(df_stat_aggregate, 1000, 28)

def steady_state_random_sampler_knee(data_set, day_min):
    data_set = data_set[data_set['day']>=day_min]
    data_set_dict = dict(list(data_set.groupby('group')))
    def boostrapper(key):
        df = data_set_dict[key]
        return df.sample(frac=1, replace=True).agg({'inter_knee_distance_adjust':['mean']})

    df_out = pd.concat([boostrapper(group) for group in list(data_set_dict.keys())], axis=0, ignore_index=True)
    df_out['group'] = list(data_set_dict.keys())
    return df_out

df_bootstrap_knee = pd.concat([steady_state_random_sampler_knee(df_stat_aggregate_knee, 28) for _ in range(1000)], axis=0, ignore_index=True)

def histogram_plotter(data_set, study_group, y_variable):
    plt.hist(data_set.loc[data_set['group']==study_group, y_variable], bins=45, color=group_2_color(study_group), alpha=0.5)
    sns.despine(left=True)

def group_mean(data_set, study_group, y_variable):
    calc_object = data_set.loc[data_set['group']==study_group, y_variable]
    return np.round(np.mean(calc_object), decimals=2)

#Iliac crest height sensitivity at steady state
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
[histogram_plotter(df_bootstrap[df_bootstrap['joint_name']=='iliac'], group, 'y') for group in ['sci', 'sci_medium', 'sci_msc']]
plt.xlabel('Iliac crest height', size=20, fontweight='bold')
plt.ylabel('Counts (n)', size=20, fontweight='bold')
ax1.annotate('SCI', xy=(3, 1), xytext=(2.92, 10), fontweight='bold', size=20)
ax1.annotate('   SCI+\nMedium', xy=(3, 1), xytext=(3.25, 10), fontweight='bold', size=20)
ax1.annotate('    SCI+\nMedium+\n   MSCs', xy=(3, 1), xytext=(3.67, 10), fontweight='bold', size=20)
ax1.annotate(group_mean(df_bootstrap[df_bootstrap['joint_name']=='iliac'], 'sci', 'y'), xy=(3, 1), xytext=(2.91, 5), fontweight='bold', size=20, color='w')
ax1.annotate(group_mean(df_bootstrap[df_bootstrap['joint_name']=='iliac'], 'sci_medium', 'y'), xy=(3, 1), xytext=(3.28, 5), fontweight='bold', size=20, color='w')
ax1.annotate(group_mean(df_bootstrap[df_bootstrap['joint_name']=='iliac'], 'sci_msc', 'y'), xy=(3, 1), xytext=(3.7, 5), fontweight='bold', size=20, color='w')
#plt.savefig('plot_iliac_crest_height_steady_state_boostrap.svg', dpi=1000)

#Inter knee distance sensitivity at steady state
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
[histogram_plotter(df_bootstrap_knee, group, 'inter_knee_distance_adjust') for group in ['sci', 'sci_medium', 'sci_msc']]
plt.xlabel('Inter knee distance', size=20, fontweight='bold')
plt.ylabel('Counts (n)', size=20, fontweight='bold')
ax1.annotate('   SCI+\nMedium+\n   MSCs', xy=(0.3, 0.05), xytext=(0.25, 6), fontweight='bold', size=20)
ax1.annotate('   SCI+\nMedium', xy=(0.3, 0.05), xytext=(0.32, 6), fontweight='bold', size=20)
ax1.annotate('SCI', xy=(0.3, 0.05), xytext=(0.42, 6), fontweight='bold', size=20)
ax1.annotate(group_mean(df_bootstrap_knee, 'sci_msc', 'inter_knee_distance_adjust'), xy=(0.3, 0.05), xytext=(0.26, 2.5), fontweight='bold', size=20, color='w')
ax1.annotate(group_mean(df_bootstrap_knee, 'sci_medium', 'inter_knee_distance_adjust'), xy=(0.3, 0.05), xytext=(0.328, 2.5), fontweight='bold', size=20, color='w')
ax1.annotate(group_mean(df_bootstrap_knee, 'sci', 'inter_knee_distance_adjust'), xy=(0.3, 0.05), xytext=(0.418, 2.5), fontweight='bold', size=20, color='w')
#plt.savefig('plot_inter_knee_distance_adjust_steady_state_boostrap.svg', dpi=1000)

#Scatterplot of bootstrapped x and y -values
#A. Pre-requisities
joint_combinations = [['iliac', 'trochanter'], ['trochanter', 'knee'], ['knee', 'ankle'], ['ankle', 'toe']]
joints = list(df_bootstrap['joint_name'].unique())
study_groups = ['sci', 'sci_msc', 'sci_medium']
df_bootstrap_summary = df_bootstrap.groupby(['joint_name', 'group'], as_index=False).mean()
#B. Plot-functions
def fancy_side_view_bootstrap_plotter(data_set, study_group):
    plot_data = data_set[data_set['group']==study_group]

    def joint_plotter(joint):
        joint_data = plot_data[plot_data['joint_name']==joint]
        plt.scatter('x', 'y', data = joint_data, color=group_2_color(study_group), alpha=0.1, marker=joint_2_shape(joint), s=10)

    [joint_plotter(joint) for joint in list(data_set['joint_name'].unique())]

    sns.despine(left=True)
    plt.xlabel('Distance [x]', size=20, fontweight='bold')
    plt.ylabel('Distance [y]', size=20, fontweight='bold')

def line_plotter(data_summary, study_group, joint_comb):
    plot_data_summary = data_summary[(data_summary['group'] == study_group) & data_summary['joint_name'].isin(joint_comb)]
    plt.plot('x', 'y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.4, linewidth=4, linestyle='--')

#C. Calling plot functions & saving
[fancy_side_view_bootstrap_plotter(df_bootstrap, group) for group in study_groups]
list(map(lambda group: list(map(lambda joint_combo: line_plotter(df_bootstrap_summary, group, joint_combo), joint_combinations)), study_groups))
#plt.savefig('plot_fancy_side_view_bootstrap.svg', dpi=1000)

def multiple_group_comparison_bootstrap(data_set, day_min, joint, variable):
    data_set = data_set[(data_set['joint_name']==joint) & (data_set['day']>=day_min)]
    data_set_dict = dict(list(data_set.groupby(['group'])))
    kruskal_raw = stats.kruskal(data_set_dict['sci'][variable], data_set_dict['sci_medium'][variable], data_set_dict['sci_msc'][variable])
    kruskal_out = np.round(kruskal_raw[1], decimals=3)
    return kruskal_out

def multiple_group_comparison_bootstrap_caller(variable):
    out = [multiple_group_comparison_bootstrap(df_stat_aggregate, 28, joint, variable) for joint in joints]
    return pd.DataFrame({'p_value' : out, 'joint_name':joints})

pd.concat([multiple_group_comparison_bootstrap_caller('x') for _ in range(1000)], axis=0, ignore_index=True)

#for Y också
#summarize
#add to plot in proper way
#add joint names to plot
#gör liknande lösning för inter knee distance









#0. Mixed ANOVA of time and group (iliac crest height and inter knee distance over time)
    #-> r2py
    #add to time overview plots
#A2. Post hoc analysis of multiple group analysis of joint position (for each joint and day separately)
    #-> tabell only
#B1. Comparing joint location over time within each group
    #add p-values for each joint at each time point
#Correlation plot between iliac crest height and inter knee distance
    #regression & causality
#Exportera tabeller