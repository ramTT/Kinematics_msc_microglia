import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import math as mt

from positioning_coordination_v2 import Position
############################################## DATA IMPORTING AND ADJUSTMENTS #############################################
#Importing data
data_set_angles = pd.read_csv('merge_side_view.csv', encoding='utf-16')
animal_key = pd.read_csv('animal_key_kinematics.csv')

#Defining column characteristics
keep_columns_angle = ['RH.index', 'side', 'day','Ang [deg].1', 'Ang [deg].2', 'Ang [deg].3', 'Ang [deg].4']
new_column_names_angle = {'Ang [deg].1':'angle_iliac_crest', 'Ang [deg].2':'angle_trochanter_major', 'Ang [deg].3':'angle_knee',
                                        'Ang [deg].4':'angle_ankle'}

#Creating instance and adjusting dataset
instance_angles = Position(data_set_angles)
instance_angles.column_adjuster(keep_columns_angle, new_column_names_angle)
instance_angles.data_frame = instance_angles.key_data_adder(animal_key, instance_angles.data_frame)

#Importing summary data for discrete joint plotting and plotting functions
from positioning_coordination_v2 import data_frame_boostrap_summary

#Summarizing angles for plotting
instance_angles.data_frame.groupby(['day', 'group'], as_index=False).mean()

#Removing unnecessary variables
instance_angles.data_frame.drop(['angle_iliac_crest', 'RH.index', 'side', 'force'], axis=1, inplace=True)

########################################### CREATING COMBINED PLOT DATASET ##########################################S
#Combining joint position & angle dataset
#A. Creating joint and angle subdata sets
joint_data = data_frame_boostrap_summary[['day', 'group', 'joint_name', 'mean_x', 'mean_y']]
angle_data = instance_angles.data_frame.loc[:, instance_angles.data_frame.columns != 'displacement'].melt(id_vars=['day', 'group'])
#B. Renaming angles
angle_data.loc[angle_data.variable=='angle_trochanter_major', 'variable'] = 'trochanter'
angle_data.loc[angle_data.variable=='angle_knee', 'variable'] = 'knee'
angle_data.loc[angle_data.variable=='angle_ankle', 'variable'] ='ankle'
#C. Renaming columns
joint_data = joint_data.rename(columns={'joint_name':'joint', 'mean_x':'x', 'mean_y':'y'})
angle_data = angle_data.rename(columns={'variable':'joint', 'value':'angle'})
#D. Adjusting angles
joint_subset = (angle_data['joint']=='trochanter') | (angle_data['joint']=='knee') | (angle_data['joint']=='ankle')
angle_data.loc[joint_subset, 'angle'] = angle_data.loc[joint_subset, 'angle'].map(lambda angle: 360-angle if angle>180 else angle)
#E. Summarizing angle data
angle_data = angle_data.groupby(['day', 'group', 'joint'], as_index=False).mean()
joint_angle_data = joint_data.set_index(['day', 'group', 'joint']).join(angle_data.set_index(['day', 'group', 'joint']))
joint_angle_data = joint_angle_data.reset_index()

#F. Calculating complementary angles (for start angle definition)
def comp_angle_calculator(df):
    joint_combinations = [['iliac', 'trochanter'], ['trochanter', 'knee'], ['knee', 'ankle']]

    def comp_angle_per_joint(joint_comb):
        horisontal_distance = df.loc[df['joint']==joint_comb[0], 'x'].reset_index()-df.loc[df['joint']==joint_comb[1], 'x'].reset_index()
        vertical_distance = df.loc[df['joint'] == joint_comb[0], 'y'].reset_index()-df.loc[
            df['joint'] == joint_comb[1], 'y'].reset_index()

        del horisontal_distance['index'], vertical_distance['index']
        hypotenusa = np.sqrt(np.float64(horisontal_distance['x'])**2 + np.float64(vertical_distance['y'])**2)

        complementary_angle = abs(mt.degrees(np.float64(np.arcsin(horisontal_distance/hypotenusa))))

        return complementary_angle

    for comb in joint_combinations:
        df.loc[df['joint']==comb[1], 'comp_angle'] = comp_angle_per_joint(comb)

    return df

#Calling function
subset_dict = dict(list(joint_angle_data.groupby(['day', 'group'])))
joint_angle_data = pd.concat([comp_angle_calculator(subset_dict[subset]) for subset in subset_dict.keys()], ignore_index=True)

#Creating empty columns for start & stop angles
joint_angle_data['start_angle'] = np.nan
joint_angle_data['stop_angle'] = np.nan
#Trochanter start & stop angle
logical_trochanter = joint_angle_data['joint']=='trochanter'
joint_angle_data.loc[logical_trochanter, 'start_angle'] = joint_angle_data.loc[logical_trochanter, 'comp_angle']+90
joint_angle_data.loc[logical_trochanter, 'stop_angle'] = joint_angle_data.loc[logical_trochanter, 'start_angle']+joint_angle_data.loc[logical_trochanter, 'angle']
#Knee start & stop angle
joint_angle_data.loc[joint_angle_data['joint']=='knee', 'stop_angle'] = 90-joint_angle_data.loc[joint_angle_data['joint']=='knee', 'comp_angle']
joint_angle_data.loc[joint_angle_data['joint']=='knee', 'start_angle'] = joint_angle_data.loc[joint_angle_data['joint']=='knee', 'stop_angle']-joint_angle_data.loc[joint_angle_data['joint']=='knee', 'angle']
#Ankle start & stop angle
joint_angle_data.loc[joint_angle_data['joint']=='ankle', 'comp_angle'] = 180-90-joint_angle_data.loc[joint_angle_data['joint']=='ankle', 'comp_angle']
joint_angle_data.loc[joint_angle_data['joint']=='ankle', 'start_angle'] = 180 + joint_angle_data.loc[joint_angle_data['joint']=='ankle', 'comp_angle']
joint_angle_data.loc[joint_angle_data['joint']=='ankle', 'stop_angle'] = joint_angle_data.loc[joint_angle_data['joint']=='ankle', 'start_angle'] + joint_angle_data.loc[joint_angle_data['joint']=='ankle', 'angle']

################################################### PLOTTING ###################################################
#Groups
study_groups = list(joint_angle_data.group.unique())
joint_list = ['trochanter', 'knee', 'ankle']

#Colors
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

#A. Wedge plots
def wedge_plot_function(axis_name, center, radius, theta1, theta2, group):
    axis_name.add_patch(patches.Wedge(center, radius, theta1, theta2, color=group_2_color(group), alpha=0.3, lw=0))

def wedge_plot(data_set, plot_day, study_group, joint):
    wedge_plot_data = data_set[(data_set['day']==plot_day) & (data_set['group']==study_group)&(data_set['joint']==joint)]

    center_x = np.float64(wedge_plot_data['x'])
    center_y = np.float64(wedge_plot_data['y'])
    angle_start = np.float64(wedge_plot_data['start_angle'])
    angle_end = np.float64(wedge_plot_data['stop_angle'])
    wedge_plot_function(ax1, (center_x, center_y), 0.5, angle_start, angle_end, study_group)
    plt.xlim([-0.25,3])
    plt.ylim([-0.25,3])
    plt.xlabel('Distance [x]', size=16, fontweight='bold')
    plt.ylabel('Distance [y]', size=16, fontweight='bold')
    plt.title('Day Post SCI'+':'+' '+str(plot_day), size=20, fontweight='bold')
    sns.despine(left=True)

#B. Joint locations & inter-joint distances
def coordinates_overtime_plot_extended_v2(data_summary, plot_day, study_group):
    #Creating datasets
    plot_data_summary = data_summary[(data_summary['day']==plot_day)&(data_summary['group']==study_group)]

    #Creating plots
    for joint in plot_data_summary['joint_name']:
        plt.scatter('mean_x', 'mean_y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.8, s=500, marker='p')

def line_plotter(data_summary, plot_day, study_group, joint_comb):
    plot_data_summary = data_summary[(data_summary['day'] == plot_day) & (data_summary['group'] == study_group)&data_summary['joint_name'].isin(joint_comb)]

    plt.plot('mean_x', 'mean_y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.2, linewidth=4, linestyle='--')

def plot_caller(plot_day):
    list(map(lambda group: coordinates_overtime_plot_extended_v2(data_frame_boostrap_summary,
                                                                 plot_day, group), study_groups))

    list(map(lambda group: list(map(lambda joint_combo: line_plotter(data_frame_boostrap_summary, plot_day, group, joint_combo), joint_combinations)), study_groups))
    plt.savefig('position_plot_'+'d'+str(plot_day)+'.jpg', dpi=1000)

#Calling all plot functions
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
list(map(lambda joint: list(map(lambda group: wedge_plot(joint_angle_data, 3, group, joint), study_groups)), joint_list))
plot_caller(3)
ax1.annotate('Trochanter\n    major', xy=(1, 1), xytext=(1.25, 2.1), fontweight='bold', size=12)
ax1.annotate('Knee', xy=(1, 1), xytext=(0.15, 0.7), fontweight='bold', size=12)
ax1.annotate('Ankle', xy=(1, 1), xytext=(1.8, 0.8), fontweight='bold', size=12)
plt.scatter(0.5, 2.9, s=100, color=group_2_color('sci'))
plt.scatter(1, 2.9, s=100, color=group_2_color('sci_medium'))
plt.scatter(1.8, 2.9, s=100, color=group_2_color('sci_msc'))
ax1.annotate('SCI', fontweight='bold', size=12, xy=(1,1), xytext=(0.55, 2.875))
ax1.annotate('SCI+Medium', fontweight='bold', size=12, xy=(1,1), xytext=(1.05, 2.875))
ax1.annotate('SCI+Medium+MSCs', fontweight='bold', size=12, xy=(1,1), xytext=(1.85, 2.875))