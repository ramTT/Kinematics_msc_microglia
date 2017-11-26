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

def group_2_marker(argument):
    switcher = {
        'sci':'*',
        'sci_medium':'+',
        'sci_msc':'o'
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

############################################################### MULTIPLE GROUP COMPARISONS ###########################################################################

#A1. Comparing joint positions between groups at each time point for each joint
def joint_comparison_between_group(data_set, plot_day, var_type):
    data_set = data_set[data_set['day']==plot_day]
    joints = list(data_set['joint_name'].unique())

    def kruskal_test(joint_name):
        data_set_dict = dict(list(data_set[data_set['joint_name']==joint_name].groupby(['group'])))
        kruskal_out = stats.kruskal(data_set_dict['sci'][var_type], data_set_dict['sci_medium'][var_type], data_set_dict['sci_msc'][var_type])
        return (plot_day, joint_name, var_type, kruskal_out[1])

    return pd.DataFrame([kruskal_test(joint) for joint in joints], columns=['day','joint', 'var_type','p_value'])

within_day_group_comp_joints_x = pd.concat([joint_comparison_between_group(df_stat_aggregate, day, 'x') for day in list(df_stat_aggregate['day'].unique())], axis=0, ignore_index=True)
within_day_group_comp_joints_y = pd.concat([joint_comparison_between_group(df_stat_aggregate, day, 'y') for day in list(df_stat_aggregate['day'].unique())], axis=0, ignore_index=True)

#A2. Comparing within group over time for each joint
def joint_comparison_within_group(data_set, study_group, var):
    calc_data = data_set[(data_set['group']==study_group)]
    def kruskal_test(joint_name):
        calc_data_sub = calc_data.loc[calc_data['joint_name']==joint_name]
        cd = dict(list(calc_data_sub.groupby(['day'])))
        kruskal_raw = stats.kruskal(cd[3][var], cd[7][var], cd[14][var], cd[21][var], cd[28][var], cd[35][var], cd[42][var])
        kruskal_out = pd.DataFrame({'joint':[joint_name], 'p_value':[kruskal_raw[1]], 'group':[study_group], 'variable':[var]})
        return kruskal_out

    return pd.concat([kruskal_test(joint) for joint in joints], axis=0, ignore_index=True)

within_group_between_day_joints_x = pd.concat([joint_comparison_within_group(df_stat_aggregate, group, 'x') for group in study_groups], axis=0, ignore_index=True)
within_group_between_day_joints_y = pd.concat([joint_comparison_within_group(df_stat_aggregate, group, 'y') for group in study_groups], axis=0, ignore_index=True)

#För inter knee distance också (between groups and within group)

def between_groups_within_time_point_knee(data_set, var):
    data_set_dict = dict(list(data_set.groupby(['day'])))
    def kruskal_test(day):
        cd = dict(list(data_set_dict[day].groupby(['group'])))
        kruskal_raw = stats.kruskal(cd['sci'][var], cd['sci_medium'][var], cd['sci_msc'][var])
        return pd.DataFrame({'day':[day], 'p_value':[kruskal_raw[1]]})

    return pd.concat([kruskal_test(day) for day in list(data_set_dict.keys())], axis=0, ignore_index=True)

within_day_between_groups_knee = between_groups_within_time_point_knee(df_stat_aggregate_knee, 'inter_knee_distance_adjust')

def within_group_over_time_knee(data_set, var):
    data_set_dict = dict(list(data_set.groupby(['group'])))
    def kruskal_test(group):
        cd = dict(list(data_set_dict[group].groupby(['day'])))
        kruskal_raw = stats.kruskal(cd[3][var], cd[7][var], cd[14][var], cd[21][var], cd[28][var], cd[35][var], cd[42][var])
        return pd.DataFrame({'group':[group], 'p_value':[kruskal_raw[1]]})

    return pd.concat([kruskal_test(group) for group in study_groups], axis=0, ignore_index=True)

within_group_between_day_knee = within_group_over_time_knee(df_stat_aggregate_knee, 'inter_knee_distance_adjust')

############################################################### POST HOC ANALYSIS OF MULTIPLE GROUP COMPARISONS ###########################################################################
import posthocs as ph

def between_group_within_time_points_post_hoc(data_set, joint, var):
    data_set = data_set[data_set['joint_name']==joint]
    data_set_dict = dict(list(data_set.groupby(['day'])))
    def pair_wise_test(day):
        pair_wise_raw = ph.posthoc_mannwhitney(data_set_dict[day], val_col=var, group_col='group')
        pair_wise_raw['index'], pair_wise_raw['day'], pair_wise_raw['joint']  = pair_wise_raw.index, day, joint
        pair_wise_raw = pair_wise_raw.melt(id_vars=['index', 'day', 'joint'])
        pair_wise_raw = pair_wise_raw[~(pair_wise_raw['index']==pair_wise_raw['variable'])]
        pair_wise_raw['index'] = pair_wise_raw['index']+'-'+pair_wise_raw['variable']
        pair_wise_raw.drop(['variable'], axis=1, inplace=True)
        pair_wise_raw.drop_duplicates(subset='value', inplace=True)
        pair_wise_raw = pair_wise_raw.rename(columns={'value':'p_value'})
        pair_wise_raw['var'] = var
        return pair_wise_raw

    return pd.concat([pair_wise_test(day) for day in list(data_set_dict.keys())], axis=0, ignore_index=True)

within_day_between_group_joints_post_hoc_x = pd.concat([between_group_within_time_points_post_hoc(df_stat_aggregate, joint, 'x') for joint in joints], axis=0, ignore_index=True)
within_day_between_group_joints_post_hoc_y = pd.concat([between_group_within_time_points_post_hoc(df_stat_aggregate, joint, 'y') for joint in joints], axis=0, ignore_index=True)

def between_group_within_time_points_post_hoc_knee(data_set, var):
    data_set_dict = dict(list(data_set.groupby(['day'])))
    def pair_wise_test(day):
        pair_wise_raw = ph.posthoc_mannwhitney(data_set_dict[day], val_col=var, group_col='group')
        pair_wise_raw['index'], pair_wise_raw['day']  = pair_wise_raw.index, day
        pair_wise_raw = pair_wise_raw.melt(id_vars=['index', 'day'])
        pair_wise_raw = pair_wise_raw[~(pair_wise_raw['index']==pair_wise_raw['variable'])]
        pair_wise_raw['index'] = pair_wise_raw['index']+'-'+pair_wise_raw['variable']
        pair_wise_raw.drop(['variable'], axis=1, inplace=True)
        pair_wise_raw.drop_duplicates(subset='value', inplace=True)
        pair_wise_raw = pair_wise_raw.rename(columns={'value':'p_value'})
        return pair_wise_raw

    return pd.concat([pair_wise_test(day) for day in list(data_set_dict.keys())], axis=0, ignore_index=True)

within_day_between_group_post_hoc_knee= between_group_within_time_points_post_hoc_knee(df_stat_aggregate_knee, 'inter_knee_distance_adjust')

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

def multiple_group_comparison_bootstrap(data_set, day_min, joint, variable):
    data_set = data_set[(data_set['joint_name']==joint) & (data_set['day']>=day_min)]
    data_set_dict = dict(list(data_set.groupby(['group'])))
    kruskal_raw = stats.kruskal(data_set_dict['sci'][variable], data_set_dict['sci_medium'][variable], data_set_dict['sci_msc'][variable])
    kruskal_out = np.round(kruskal_raw[1], decimals=3)
    return kruskal_out

def multiple_group_comparison_bootstrap_caller(variable):
    out = [multiple_group_comparison_bootstrap(df_stat_aggregate, 28, joint, variable) for joint in joints]
    return pd.DataFrame({'p_value' : out, 'joint_name':joints})

#Bootstrapping p-values for multiple group comparison
multiple_group_p_value_x = pd.concat([multiple_group_comparison_bootstrap_caller('x') for _ in range(1000)], axis=0, ignore_index=True)
multiple_group_p_value_y = pd.concat([multiple_group_comparison_bootstrap_caller('y') for _ in range(1000)], axis=0, ignore_index=True)

p_value_x = multiple_group_p_value_x.groupby(['joint_name']).mean()
p_value_y = multiple_group_p_value_y.groupby(['joint_name']).mean()

#Post hoc test in steady state (boostrapped)
def steady_state_joint_post_hoc(data_set, joint, min_day, var):
    data_set = data_set.loc[(data_set['joint_name']==joint) & (data_set['day']>=min_day)]

    def post_hoc_test():
        pair_wise_raw = ph.posthoc_mannwhitney(data_set, val_col=var, group_col='group')
        pair_wise_raw['index'] = pair_wise_raw.index
        pair_wise_raw = pair_wise_raw.melt(id_vars=['index'])
        pair_wise_raw = pair_wise_raw[~(pair_wise_raw['index'] == pair_wise_raw['variable'])]
        pair_wise_raw['index'] = pair_wise_raw['index'] + '-' + pair_wise_raw['variable']
        pair_wise_raw.drop(['variable'], axis=1, inplace=True)
        pair_wise_raw.drop_duplicates(subset='value', inplace=True)
        pair_wise_raw = pair_wise_raw.rename(columns={'value': 'p_value'})
        pair_wise_raw['variable'] = var
        pair_wise_raw['joint'] = joint
        return pair_wise_raw

    pair_wise_boot_data = pd.concat([post_hoc_test() for _ in range(100)], axis=0, ignore_index=True)
    return pair_wise_boot_data

post_hoc_joint_x = pd.concat([steady_state_joint_post_hoc(df_stat_aggregate, joint, 28, 'x') for joint in joints], axis=0, ignore_index=True)
post_hoc_joint_y = pd.concat([steady_state_joint_post_hoc(df_stat_aggregate, joint, 28, 'y') for joint in joints], axis=0, ignore_index=True)

post_hoc_steady_state_mean_joints_x = post_hoc_joint_x.groupby(['index', 'variable', 'joint'], as_index=False).mean()
post_hoc_steady_state_mean_joints_y = post_hoc_joint_y.groupby(['index', 'variable', 'joint'], as_index=False).mean()

def steady_state_joint_post_hoc_knee(data_set, min_day):
    data_set_calc = data_set[data_set['day']>=min_day]
    def post_hoc_test():
        pair_wise_raw = ph.posthoc_mannwhitney(data_set_calc, val_col='inter_knee_distance_adjust', group_col='group')
        pair_wise_raw['index'] = pair_wise_raw.index
        pair_wise_raw = pair_wise_raw.melt(id_vars=['index'])
        pair_wise_raw = pair_wise_raw[~(pair_wise_raw['index'] == pair_wise_raw['variable'])]
        pair_wise_raw['index'] = pair_wise_raw['index'] + '-' + pair_wise_raw['variable']
        pair_wise_raw.drop(['variable'], axis=1, inplace=True)
        pair_wise_raw.drop_duplicates(subset='value', inplace=True)
        pair_wise_raw = pair_wise_raw.rename(columns={'value': 'p_value'})
        return pair_wise_raw

    return pd.concat([post_hoc_test() for _ in range(1000)], axis=0, ignore_index=True)

post_hoc_joint_knee = steady_state_joint_post_hoc_knee(df_stat_aggregate_knee, 28)
post_hoc_joint_knee_mean = post_hoc_joint_knee.groupby(['index']).mean()

#Dictionaries for adding p-values and joint names to plot
position_x_dictionary = {'Ankle':[2.7, 1.6], 'Iliac crest':[0.05, 4.1], 'Knee':[0.2, 1.1], 'Toe':[1.6, 0.45], 'Trochanter major':[2.2, 3.4]}
position_y_dictionary = {'Ankle':[2.7, 1.5], 'Iliac crest':[0.05, 4], 'Knee':[0.2, 1.0], 'Toe':[1.6, 0.35], 'Trochanter major':[2.2, 3.3]}
position_name_dictionary = {'Ankle':[2.7, 1.7], 'Iliac crest':[0.05, 4.2], 'Knee':[0.2, 1.2], 'Toe':[1.6, 0.55], 'Trochanter major':[2.2, 3.5]}

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

    for joint, pos in enumerate(position_x_dictionary):
        if(np.float(p_value_x.iloc[joint,:])< 0.05):
            color_x = 'g'
        else:
            color_x='r'
        if(np.float(p_value_y.iloc[joint,:])<0.05):
            color_y = 'g'
        else:
            color_y = 'r'

        plt.annotate('X: '+str('{:03.3f}'.format(np.round(np.float(p_value_x.iloc[joint,:]), 3))), xy=(1,1), xytext=(position_x_dictionary[pos][0], position_x_dictionary[pos][1]), fontweight='bold', size=10, color=color_x)
        plt.annotate('Y: '+str('{:03.3f}'.format(np.round(np.float(p_value_y.iloc[joint,:]), 3))), xy=(1,1), xytext=(position_y_dictionary[pos][0], position_y_dictionary[pos][1]), fontweight='bold', size=10, color=color_y)
        plt.annotate(str(pos), xy=(1,1), xytext=(position_name_dictionary[pos][0], position_name_dictionary[pos][1]), fontweight='bold', size=14)

def line_plotter(data_summary, study_group, joint_comb):
    plot_data_summary = data_summary[(data_summary['group'] == study_group) & data_summary['joint_name'].isin(joint_comb)]
    plt.plot('x', 'y', data=plot_data_summary, color=group_2_color(study_group), linewidth=4, linestyle='--')

#C. Calling plot functions & saving
[fancy_side_view_bootstrap_plotter(df_bootstrap, group) for group in study_groups]
list(map(lambda group: list(map(lambda joint_combo: line_plotter(df_bootstrap_summary, group, joint_combo), joint_combinations)), study_groups))
#plt.savefig('plot_fancy_side_view_bootstrap.svg', dpi=1000)

########################################################################### KNEE PLOT ###################################################################################
#Fancy plot with bootstrap of inter knee distance

#Adding a y-value with custom buit jitter for plotting purposes
df_bootstrap_knee['y'] = df_bootstrap_knee['y'].map(lambda y_value: np.random.uniform(0.95, 1.05))
#Creating a crotch point dataset for plotting
df_line_data_knee = df_bootstrap_knee.iloc[:,~(df_bootstrap_knee.columns=='y')].groupby(['group']).mean()
df_line_data_knee = df_line_data_knee.rename(columns={'inter_knee_distance_adjust':'x_mean'})
df_line_data_knee['y_mean'] = 1
df_line_data_knee['x_crotch'] = df_line_data_knee['x_mean'] / 2
df_line_data_knee['y_crotch'] = 2.5
df_line_data_knee['x_fix'] = 0
df_line_data_knee['y_fix'] = 1
df_line_data_knee = df_line_data_knee.reset_index()
#From wide to long format for fixing
df_line_data_knee = df_line_data_knee.melt(id_vars=['group'])
df_line_data_knee['coord'] = df_line_data_knee['variable'].map(lambda var: var[0])
df_line_data_knee['variable'] = df_line_data_knee['variable'].map(lambda var: var[2:])
#Back to wide from long to fix coordinates
df_line_data_knee = pd.pivot_table(df_line_data_knee, values='value', index=['group', 'variable'], columns ='coord').reset_index()


#Knee - p-value for multiple group comparison
def knee_multiple_group_comp(dataset, min_day):
    calc_data = dataset[dataset['day']>=min_day]
    calc_data_dict = dict(list(calc_data.groupby(['group'])))
    calc_data_boot = pd.concat([calc_data_dict[group].sample(frac=1, replace=True) for group in calc_data_dict], axis=0, ignore_index=True)
    kruskal_raw = stats.kruskal(calc_data_boot.loc[calc_data_boot['group']=='sci', 'inter_knee_distance_adjust'], calc_data_boot.loc[calc_data_boot['group']=='sci_medium', 'inter_knee_distance_adjust'], calc_data_boot.loc[calc_data_boot['group']=='sci_msc', 'inter_knee_distance_adjust'])
    p_value = np.round(np.float(kruskal_raw[1]), 3)

    return p_value

knee_kruskal_p = pd.DataFrame({'p_value':[knee_multiple_group_comp(df_stat_aggregate_knee, 28) for _ in range(1000)]})


def inter_knee_distance_plot(data_bootstrap, data_line, study_group):
    plot_data_bootstrap = data_bootstrap[(data_bootstrap['group']==study_group)]
    plot_data_line = data_line[data_line['group']==study_group]
    #Dots
    plt.scatter('inter_knee_distance_adjust', 'y', data=plot_data_bootstrap, color=group_2_color(study_group), alpha=0.05)
    #Lines
    plt.plot('x', 'y', data = plot_data_line[~(plot_data_line['variable'] == 'mean')], color = group_2_color(study_group), linewidth = 4, linestyle = '--')
    plt.plot('x', 'y', data = plot_data_line[~(plot_data_line['variable'] == 'fix')], color = group_2_color(study_group), linewidth = 4, linestyle = '--')

    #Plot adjust
    sns.despine(left=True)
    plt.xlabel('Distance [x]', size=15, fontweight='bold')
    plt.xticks(list(np.arange(0, 0.6, 0.05)))
    plt.yticks(list(np.arange(0, 3, 0.25)))
    plt.ylim([0.9, 2.6])
    plt.xlim([0, 0.55])

[inter_knee_distance_plot(df_bootstrap_knee, df_line_data_knee, group) for group in study_groups]
#plt.savefig('plot_fancy_bottom_view_bootstrap.svg', dpi=1000)

########################################################################### CORRELATION PLOT ##############################################################################
#Correlation between inter knee distance, height (iliac crest & trochanter major)

correlation_data = df_stat_aggregate_knee.merge(df_stat_aggregate[(df_stat_aggregate['joint_name'].isin(['iliac', 'trochanter']))].drop(['x'], axis=1), on=['RH.index', 'day', 'group'])

def correlation_plotter(data_set, study_group, height_type, y_variable):
    plot_data = data_set[(data_set['joint_name']==height_type)&(data_set['group']==study_group)]

    plt.scatter('y','inter_knee_distance_adjust', data=plot_data, color=group_2_color(study_group), marker=group_2_marker(study_group), s=100)
    sns.despine(left=True)
    plt.ylabel('Inter knee distance', fontweight='bold', size=20)
    plt.xlabel(y_variable, fontweight='bold', size=20)

[correlation_plotter(correlation_data, group, 'iliac', 'Iliac crest height') for group in study_groups]
#plt.savefig('correlation_plot.svg', dpi=1000)

#Evaluating normal distribution within variable and group
[stats.normaltest(correlation_data.loc[correlation_data['group']==group,'inter_knee_distance_adjust']) for group in study_groups]
[stats.normaltest(correlation_data.loc[correlation_data['group']==group,'y']) for group in study_groups]
# -> normality is not fulfilled, using non-parametric tests

#Estimating correlations with spearman and kendall
def correlation_builder(data_set):

    def correlation_sub(joint_type):
        calc_data = data_set.loc[(data_set['joint_name']==joint_type)]
        p_value_spearman = [np.round(np.float64(stats.spearmanr(calc_data.loc[calc_data['group']==sub_group, ['inter_knee_distance_adjust', 'y']])[1]), 3) for sub_group in study_groups]
        p_value_kendall = [np.round(np.float64(stats.kendalltau(calc_data.loc[calc_data['group']==sub_group, 'inter_knee_distance_adjust'], calc_data.loc[calc_data['group']==sub_group, 'y'])[1]),3) for sub_group in study_groups]
        out_df = pd.DataFrame({'spearman_p':p_value_spearman, 'kendall_p':p_value_kendall, 'joint':joint_type, 'group':study_groups})
        return out_df
    return pd.concat([correlation_sub('iliac'), correlation_sub('trochanter')], axis=0)

correlation_builder(correlation_data)

###Exporting tables as CSV
#1. Within group within days joints
pd.concat([within_day_group_comp_joints_x, within_day_group_comp_joints_y], axis=0, ignore_index=True).to_csv('within_day_group_comp_joints.csv')
#2. Within group between day joints
pd.concat([within_group_between_day_joints_x, within_group_between_day_joints_y], axis=0, ignore_index=True).to_csv('within_group_between_days_joints.csv')
#3. Within day between groups knee
within_day_between_groups_knee.to_csv('within_day_between_group_knee.csv')
#4. Within group between day knee
within_group_between_day_knee.to_csv('within_group_between_day_knee.csv')
#5. Within day between group post hoc joints
pd.concat([within_day_between_group_joints_post_hoc_x, within_day_between_group_joints_post_hoc_y], axis=0, ignore_index=).to_csv('within_day_between_group_post_hoc_joints.csv')
#6. Within day between group post hoc knee
within_day_between_group_post_hoc_knee.to_csv('within_day_between_group_post_hoc_knee.csv')
#7. Steady state multiple group test joints
p_value_x['var'], p_value_x['joint_name'] = 'x', p_value_x.index
p_value_y['var'], p_value_y['joint_name'] = 'y', p_value_y.index
pd.concat([p_value_x, p_value_y], axis=0, ignore_index=True).to_csv('steady_state_multiple_group_test_joints.csv')
#8. Steady state post hoc test joints
pd.concat([post_hoc_steady_state_mean_joints_x, post_hoc_steady_state_mean_joints_y], axis=0, ignore_index=True).to_csv('steady_state_post_hoc_test_joints.csv')
#9. Steady state multiple group test knee
knee_kruskal_p.mean().to_csv('steady_state_multiple_group_test_knee.csv')
#10. Steady state post hoc test knee
post_hoc_joint_knee_mean.to_csv('steady_state_post_hoc_test_knee.csv')
#11. P-values for spearman & kendall correlation coefficients
correlation_builder(correlation_data).to_csv('correlation_p_values.csv')