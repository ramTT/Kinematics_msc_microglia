import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from positioning_coordination_v2 import Position
from positioning_coordination_v2 import group_2_color

#Importing data
data_set_knee = pd.read_csv('merge_bottom_view.csv')
animal_key = pd.read_csv('animal_key_kinematics.csv')

#Creating variables for data adjustment
keep_columns_knee = ['Dist [cm].1', 'RH.index', 'day']
new_column_names_knee = {'Dist [cm].1': 'inter_knee_distance'}

#Creating instance and adjusting data
instance_knee = Position(data_set_knee)
instance_knee.column_adjuster(keep_columns_knee, new_column_names_knee)
instance_knee.data_frame = instance_knee.key_data_adder(animal_key, instance_knee.data_frame)
del instance_knee.data_frame['force']

#Adjusting for displacement
displacement_min = min(instance_knee.data_frame['displacement'])
instance_knee.data_frame['displacement'] /=displacement_min
instance_knee.data_frame.loc[instance_knee.data_frame['day']==3,['displacement']] = 1
instance_knee.data_frame['inter_knee_distance_adjust'] = instance_knee.data_frame['inter_knee_distance']*instance_knee.data_frame['displacement']
instance_knee.data_frame['inter_knee_distance_adjust'] = 1/instance_knee.data_frame['inter_knee_distance_adjust']

#Aggregating data
data_knee_aggregate = instance_knee.data_frame.groupby(['RH.index', 'day', 'group'], as_index=False).agg({'inter_knee_distance_adjust':['mean']})
data_knee_aggregate.columns = ['RH.index', 'day', 'group', 'inter_knee_distance_adjust']
data_knee_summary = data_knee_aggregate.groupby(['day', 'group'], as_index=False).agg({'inter_knee_distance_adjust':['mean']})
data_knee_summary.columns = ['day', 'group', 'inter_knee_distance_adjust']

#Plot preparations
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]
#List of groups
study_groups = ['sci', 'sci_medium', 'sci_msc']
#Creating x-variable (for origo) and y-variable (for vertical separation)
instance_knee.data_frame['x'], data_knee_aggregate['x'], data_knee_summary['x'] = [0, 0, 0]
instance_knee.data_frame['y'], data_knee_aggregate['y'], data_knee_summary['y'] = [1, 1, 1]
#Adjusting y-values
def y_adjustor(dataset):
    dataset.loc[dataset['group'] == 'sci', 'y'] = 1.25
    dataset.loc[dataset['group'] == 'sci_medium', 'y'] = 0.75

list(map(lambda data_set: y_adjustor(data_set), [instance_knee.data_frame, data_knee_aggregate, data_knee_summary]))
#Creating an artifical crotch point
data_knee_summary['x_artificial'] = data_knee_summary['inter_knee_distance_adjust']/2
data_knee_summary['y_artificial'] = data_knee_summary['y']+2

#Optimize code!!!!
temp = data_knee_summary.melt(id_vars=['day', 'group', 'inter_knee_distance_adjust'])
temp['group'] = temp['group']+'_'+temp['variable'].map(lambda name: name[2:])
temp['variable'] = temp['variable'].map(lambda name: name[0])
temp = temp.pivot_table(values='value', columns='variable', index=['day', 'group', 'inter_knee_distance_adjust']).reset_index()

temp.loc[temp['group']=='sci_artificial', 'group'] = 'sci_'
temp.loc[temp['group']=='sci_medium_artificial', 'group'] = 'sci_medium_'
temp.loc[temp['group']=='sci_msc_artificial', 'group'] = 'sci_msc_'

temp.loc[temp['group']=='sci_', 'group'] = 'sci'
temp.loc[temp['group']=='sci_medium_', 'group'] = 'sci_medium'
temp.loc[temp['group']=='sci_msc_', 'group'] = 'sci_msc'

temp2 = temp.copy()
temp2.loc[temp2['x']==0, 'x'] = temp2.loc[temp2['x']==0, 'inter_knee_distance_adjust']
#!!!!


#Plotting
def inter_knee_distance_plot(data_technical, data_biological, data_summary, study_group, plot_day):
    #Creating plot data
    plot_data_technical = data_technical[(data_technical['group']==study_group)&(data_technical['day']==plot_day)]
    plot_data_biological = data_biological[(data_biological['group']==study_group)&(data_biological['day']==plot_day)]
    plot_data_summary = data_summary[(data_summary['group']==study_group)&(data_summary['day']==plot_day)]

    #Creating plots
    #A. Points
    plt.scatter('inter_knee_distance_adjust', 'y', data = plot_data_technical, color=group_2_color(study_group), alpha=0.2, s=50)
    plt.scatter('inter_knee_distance_adjust', 'y', data=plot_data_biological, color=group_2_color(study_group), alpha=0.5,
                marker="^", s=200)
    plt.scatter('inter_knee_distance_adjust', 'y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.5, s=1000,
                marker="p")

    plt.scatter('x', 'y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.5, s=1000, marker="p")
    #B. Crotch point
    plt.scatter('x_artificial', 'y_artificial', data=plot_data_summary, color=group_2_color(study_group), alpha=0.5, s=1000, marker="p")

    #Plot adjust
    sns.despine(left=True)
    plt.xlabel('Distance [x]', size=15, fontweight='bold')
    plt.xticks(list(np.arange(0, 4.5, 0.25)))
    plt.yticks(list(np.arange(0, 4.5, 0.5)))
    plt.title('Day (Post SCI):'+' '+str(plot_day), fontweight='bold', size=20)
    plt.ylim([0, 4])
    plt.xlim([-0.1, 0.9])

def line_plotter(data_set, study_group, plot_day):
    plot_data_set = data_set[(data_set['group']==study_group)&(data_set['day']==plot_day)]
    plt.plot('x', 'y', data = plot_data_set, lw=5, alpha=0.7, color=group_2_color(study_group))

def plot_caller(plotDay):
    list(map(lambda group: inter_knee_distance_plot(instance_knee.data_frame, data_knee_aggregate, data_knee_summary, group, plotDay), study_groups))
    list(map(lambda group: line_plotter(temp, group, plotDay), ['sci', 'sci_medium', 'sci_msc']))
    list(map(lambda group: line_plotter(temp2, group, plotDay), ['sci', 'sci_medium', 'sci_msc']))
    plt.savefig('plot_bottom_d'+str(plotDay)+'.svg', dpi=1000)
    #plt.clf()

plot_caller(7)
#Saving figs
#[plot_caller(day) for day in list(instance_knee.data_frame['day'].unique())]