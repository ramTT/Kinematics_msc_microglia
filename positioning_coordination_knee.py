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

#Aggregating data
data_knee_aggregate = instance_knee.data_frame.groupby(['RH.index', 'day', 'group'], as_index=False).agg({'inter_knee_distance_adjust':['mean']})
data_knee_aggregate.columns = ['RH.index', 'day', 'group', 'inter_knee_distance_adjust']
data_knee_summary = data_knee_aggregate.groupby(['day', 'group'], as_index=False).agg({'inter_knee_distance_adjust':['mean']})
data_knee_summary.columns = ['day', 'group', 'inter_knee_distance_adjust']

#Plot preparations
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

study_groups = ['sci', 'sci_medium', 'sci_msc']

instance_knee.data_frame['y'] = 1
data_knee_aggregate['y'] = 1
data_knee_summary['y'] = 1

#Plotting
def inter_knee_distance_plot(data_technical, data_biological, data_summary, study_group, plot_day):
    #Creating plot data
    plot_data_technical = data_technical[(data_technical['group']==study_group)&(data_technical['day']==plot_day)]
    plot_data_biological = data_biological[(data_biological['group']==study_group)&(data_biological['day']==plot_day)]
    plot_data_summary = data_summary[(data_summary['group']==study_group)&(data_summary['day']==plot_day)]

    #Creating plots
    plt.scatter('inter_knee_distance_adjust', 'y', data = plot_data_technical, color=group_2_color(study_group), alpha=0.2, s=50)
    plt.scatter('inter_knee_distance_adjust', 'y', data=plot_data_biological, color=group_2_color(study_group), alpha=0.5,
                marker="^", s=200)
    plt.scatter('inter_knee_distance_adjust', 'y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.5, s=1000,
                marker="p")

    #Plot adjust
    sns.despine(left=True)
    plt.xlabel('Distance (x)', size=15, fontweight='bold')
    plt.xticks(list(np.arange(0, 4.5, 0.5)))
    plt.yticks(list(np.arange(0, 3.5, 0.5)))

list(map(lambda group: inter_knee_distance_plot(instance_knee.data_frame, data_knee_aggregate, data_knee_summary, group, 3), study_groups))

#Skapa en distans av bredden -> dvs plotta två punkter per punkt
#Gör funktion som plottar tredje punkt ovan mittpunkten på fixed höjd.
#Justera storlek med avseende på osäkerhet (hämta in std funktion från tidigare skript?)