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


from positioning_coordination_knee import instance_knee, data_knee_aggregate, data_knee_summary
#Raw - knee
df_stat_raw_knee = instance_knee.data_frame.copy()
df_stat_raw_knee.head()


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


def histogram_plotter(data_set, study_group):
    plt.hist(data_set.loc[(data_set['joint_name']=='iliac')&
                              (data_set['group']==study_group), 'y'],
             bins=45, color=group_2_color(study_group), alpha=0.5)
    sns.despine(left=True)

def group_mean(data_set, study_group):
    calc_object = data_set.loc[(data_set['group']==study_group)&(data_set['joint_name']=='iliac'), 'y']
    return np.round(np.mean(calc_object), decimals=2)

#Iliac crest height sensitivity at steady state
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
[histogram_plotter(df_bootstrap, group) for group in ['sci', 'sci_medium', 'sci_msc']]
plt.xlabel('Iliac crest height', size=20, fontweight='bold')
plt.ylabel('Counts (n)', size=20, fontweight='bold')
ax1.annotate('SCI', xy=(3, 1), xytext=(2.9, 10), fontweight='bold', size=10)
ax1.annotate('     SCI+\nMedium', xy=(3, 1), xytext=(3.25, 10), fontweight='bold', size=10)
ax1.annotate('   SCI+\nMedium+\n   MSCs', xy=(3, 1), xytext=(3.65, 10), fontweight='bold', size=10)
ax1.annotate(group_mean(df_bootstrap, 'sci'), xy=(3, 1), xytext=(2.89, 5), fontweight='bold', size=10, color='w')
ax1.annotate(group_mean(df_bootstrap, 'sci_medium'), xy=(3, 1), xytext=(3.28, 5), fontweight='bold', size=10, color='w')
ax1.annotate(group_mean(df_bootstrap, 'sci_msc'), xy=(3, 1), xytext=(3.69, 5), fontweight='bold', size=10, color='w')

#Inter knee distance sensitivity at steady state




#Add density lines (without fill) to plot
#More ways to display boostrapped data? -> evaluate x value
#Scatterplot of bootstrapped x and y -values -> how do they cluster?
    #Blir mean för resp grupp men ok -> dra fler replikat om krävs
    #Plotta 1 grupp och joint åt gången -> olika färg för grupp och olika mönster för joint
    #Maxa denna plot
    #= steady state mean fancy side view plot. Add lines to mean of mean values.
    #samma för inter knee distance






#0. Mixed ANOVA of time and group (iliac crest height and inter knee distance over time)
    #-> r2py
    #add to time overview plots
#A2. Post hoc analysis of multiple group analysis of joint position (for each joint and day separately)
    #-> python
#B1. Comparing joint location over time within each group
    #add p-values for each joint at each time point
#Correlation plot between iliac crest height and inter knee distance
#Exportera tabeller