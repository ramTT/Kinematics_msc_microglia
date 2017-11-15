import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

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
joint_data = data_frame_boostrap_summary[['day', 'group', 'joint_name', 'mean_x', 'mean_y']]
angle_data = instance_angles.data_frame.loc[:, instance_angles.data_frame.columns != 'displacement'].melt(id_vars=['day', 'group'])

angle_data.loc[angle_data.variable=='angle_trochanter_major', 'variable'] = 'trochanter'
angle_data.loc[angle_data.variable=='angle_knee', 'variable'] = 'knee'
angle_data.loc[angle_data.variable=='angle_ankle', 'variable'] ='ankle'

joint_data = joint_data.rename(columns={'joint_name':'joint', 'mean_x':'x', 'mean_y':'y'})
angle_data = angle_data.rename(columns={'variable':'joint', 'value':'angle'})
angle_data = angle_data.groupby(['day', 'group', 'joint'], as_index=False).mean()

joint_angle_data = joint_data.set_index(['day', 'group', 'joint']).join(angle_data.set_index(['day', 'group', 'joint']))
joint_angle_data = joint_angle_data.reset_index()

#Creating plot functions
study_groups = list(joint_angle_data.group.unique())

#A. Wedge plots
def wedge_plot_function(axis_name, center, radius, theta1, theta2, group_color):
    axis_name.add_patch(patches.Wedge(center, radius, theta1, theta2, color=group_color))


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

wedge_plot_function(ax1, (1, 1), 0.5, 180, 270, 'r')
plt.xlim([0,3])
plt.ylim([0,3])
sns.despine(left=True)





