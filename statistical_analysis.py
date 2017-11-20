import pandas as pd
import numpy as np
import scipy.stats as stats

#Importing datasets and adjusting
from positioning_coordination_v2 import data_frame_boostrap_raw, data_frame_boostrap_aggregate, data_frame_boostrap_summary

#Raw
df_stat_raw = data_frame_boostrap_raw.copy()
df_stat_raw = df_stat_raw.rename(columns={'x_value':'x', 'y_value':'y'})
#Aggregate
df_stat_aggregate = data_frame_boostrap_aggregate.copy()
df_stat_aggregate = df_stat_aggregate.rename(columns={'mean_x':'x', 'mean_y':'y'})
df_stat_aggregate.drop(['std_x', 'std_y'], axis=1, inplace=True)
#Summary
df_stat_summary = data_frame_boostrap_summary.copy()
df_stat_summary.drop(['std_x', 'std_y', 'std_norm'], axis=1, inplace=True)
df_stat_summary = df_stat_summary.rename(columns={'mean_x':'x', 'mean_y':'y'})

#Mixed ANOVA of time and group (iliac crest height and inter knee distance over time)
    #Post hoc analysis 1: between group per time point -> DONE
    #Post hoc analysis 2: within group over time
#Steady state analysis (iliac crest height and inter knee distance)
    #Sensitivity analysis

#A1. Comparing joint positions between groups at each time point for each joint
def joint_comparison_between_group(data_set, plot_day):
    data_set = data_set[data_set['day']==plot_day]
    joints = list(data_set['joint_name'].unique())

    def kruskal_test(joint_name):
        data_set_dict = dict(list(data_set[data_set['joint_name']==joint_name].groupby(['group'])))
        kruskal_out = stats.kruskal(data_set_dict['sci']['x'], data_set_dict['sci_medium']['x'], data_set_dict['sci_msc']['x'])
        return (plot_day, joint_name, kruskal_out[1])

    return pd.DataFrame([kruskal_test(joint) for joint in joints], columns=['day','joint', 'p_value'])

pd.concat([joint_comparison_between_group(df_stat_aggregate, day) for day in list(df_stat_aggregate['day'].unique())], axis=0, ignore_index=True)

#A2. Post hoc analysis of multiple group analysis of joint position (for each joint and day separately)