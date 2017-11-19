import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Position:

    def __init__(self, data_frame):
        self.data_frame = data_frame

    def column_adjuster(self, keep_cols, new_column_names):
        '''Selects relevant columns and renames them'''
        self.data_frame = self.data_frame[keep_cols]
        self.data_frame = self.data_frame.rename(columns = new_column_names)

    @staticmethod
    def key_data_adder(key_data, data_set):
        '''Merges dataset with additional info about the replicates'''
        if(data_set['RH.index'].dtypes != 'int64'):
            data_set['RH.index'] = data_set['RH.index'].map(lambda rh_index: rh_index[:3])
            data_set['RH.index'] = data_set['RH.index'].astype('int64')

        data_set = data_set.merge(key_data, on='RH.index')
        return data_set

    def coordinate_normalizer(self, coord_type):
        '''Normalizes x-and y coordinates to zero-base'''
        adjust_cols = [col_name for col_name in self.data_frame.columns if col_name[0] == coord_type]
        unadjusted_dict = dict(list(self.data_frame.groupby('side')))
        unadjusted_dict['left'] = unadjusted_dict['left'][adjust_cols].sub(
            unadjusted_dict['left'][coord_type + '_origo'], axis=0)
        unadjusted_dict['right'] = unadjusted_dict['right'][adjust_cols].sub(
            unadjusted_dict['right'][coord_type + '_origo'], axis=0) * (-1)

        self.data_frame[adjust_cols] = pd.concat(unadjusted_dict, ignore_index=True)

    def melting_updating_casting(self):
        data_to_melt = self.data_frame.drop(['side', 'force', 'displacement'], axis=1)
        data_melt = data_to_melt.melt(id_vars=['RH.index', 'day', 'group'])

        data_melt['coord_type'] = data_melt['variable'].apply(lambda joint: joint[0])
        data_melt['joint_name'] = data_melt['variable'].apply(lambda joint: joint[2:])
        self.data_frame_melt = data_melt
        #del data_melt['variable']

        #self.data_frame_melt = data_melt.pivot_table(index=['RH.index', 'day', 'group', 'joint_name'],
        #                                 values='value', columns='coord_type').reset_index()

    def adjustor(self, adjust_var):
        self.data_frame_melt[adjust_var + '_adjust'] = self.data_frame_melt[adjust_var] * \
                                                                    self.data_frame_melt['displacement']

#1. Importing data
data_set_position = pd.read_csv('merge_side_view.csv', encoding='utf-16')
animal_key = pd.read_csv('animal_key_kinematics.csv')

#2. Data adjustments
#A. Importing data, creating variables
keep_columns_position = [': x [cm]', 'y [cm]', 'Joint 2: x [cm]', 'y [cm].1',
                'Joint 3: x [cm]', 'y [cm].2', 'Joint 4: x [cm]', 'y [cm].3', 'Joint 5: x [cm]', 'y [cm].4',
                'Joint 6: x [cm]', 'y [cm].5', 'RH.index', 'day', 'side']

new_column_names_position = {': x [cm]':'x_origo', 'y [cm]':'y_origo', 'Joint 2: x [cm]':'x_iliac', 'y [cm].1':'y_iliac',
                    'Joint 3: x [cm]':'x_trochanter', 'y [cm].2':'y_trochanter', 'Joint 4: x [cm]':'x_knee',
                    'y [cm].3':'y_knee', 'Joint 5: x [cm]':'x_ankle', 'y [cm].4':'y_ankle',
                    'Joint 6: x [cm]':'x_toe', 'y [cm].5':'y_toe'}

#B. Creating instance, calling methods
instance_position = Position(data_set_position)
instance_position.column_adjuster(keep_columns_position, new_column_names_position)
instance_position.data_frame = instance_position.key_data_adder(animal_key, instance_position.data_frame)

instance_position.coordinate_normalizer('x')
instance_position.coordinate_normalizer('y')

#C. Adjusting negative y-values in sci-group
y_columns = [col_name for col_name in instance_position.data_frame.columns if col_name[0]=='y']
instance_position.data_frame[y_columns] = instance_position.data_frame[y_columns].applymap(lambda value: value*(-1) if value<0 else value)

#################################################################### BOOTSTRAPPING DATA FOR PLOTTING ##############################################################
#A. Bootstrapping data from each animal/day/variable
data_frame_boostrap = instance_position.data_frame.drop(['side', 'force', 'displacement'], axis=1)
data_frame_boostrap = data_frame_boostrap.melt(id_vars = ['RH.index', 'day', 'group'])
data_frame_boostrap = dict(list(data_frame_boostrap.groupby(['variable', 'RH.index', 'day'])))

data_frame_boostrap = pd.concat([data_frame_boostrap[key].sample(n=20, replace=True) for key in data_frame_boostrap.keys()], axis=0, ignore_index=True)
data_frame_boostrap['coord_type'] = data_frame_boostrap['variable'].map(lambda word: word[0])
data_frame_boostrap['joint_type'] = data_frame_boostrap['variable'].map(lambda word: word[2:])
del data_frame_boostrap['variable']

#B. Creating dataset of bootstrapped technical replicates in long format
data_frame_boostrap_raw = data_frame_boostrap[data_frame_boostrap['coord_type']=='x']
data_frame_boostrap_raw = data_frame_boostrap_raw.rename(columns={'value':'x_value'})
del data_frame_boostrap_raw['coord_type']
data_frame_boostrap_raw = data_frame_boostrap_raw.assign(y_value=data_frame_boostrap[data_frame_boostrap['coord_type']=='y']['value'].values)

#C. Agreggating bootstrap data for each animal/day/joint_type
data_frame_boostrap_aggregate = data_frame_boostrap_raw.groupby(['RH.index', 'day', 'joint_type'], as_index=False).agg(['mean', 'std']).reset_index()
data_frame_boostrap_aggregate.columns = ['RH.index', 'day', 'joint_name', 'mean_x', 'std_x', 'mean_y', 'std_y']

#D. Summarizing boostrap data for each group/day/joint
#Adding back group (from animal key dataset)
data_frame_boostrap_aggregate = data_frame_boostrap_aggregate.merge(animal_key[['RH.index', 'group']], on ='RH.index')

def mean_of_std(lambda_var):
    return np.sqrt(np.divide(np.sum(lambda_var**2), lambda_var.nunique()))

data_frame_boostrap_summary = data_frame_boostrap_aggregate.groupby(['day', 'group', 'joint_name'],
    as_index=False).agg({'mean_x':'mean','mean_y':'mean','std_x':lambda std_dev: mean_of_std(std_dev),
                         'std_y': lambda std_dev: mean_of_std(std_dev)})


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

#Function calls both plot functions
joint_combinations = [['iliac', 'trochanter'], ['trochanter', 'knee'], ['knee', 'ankle'], ['ankle', 'toe']]
study_groups = ['sci', 'sci_medium', 'sci_msc']

######################################################### PLOTTING ALL JOINTS OVER TIME  ##########################################################
def coordinates_overtime_plot_extended_v2(data_technical, data_biological, data_summary, plot_day, study_group):
    #Creating datasets
    plot_data_technical = data_technical[(data_technical['day']==plot_day) & (data_technical['group']==study_group)]
    plot_data_biological = data_biological[(data_biological['day']==plot_day)&(data_biological['group']==study_group)]
    plot_data_summary = data_summary[(data_summary['day']==plot_day)&(data_summary['group']==study_group)]

    #Creating plots
    plt.scatter('x_value', 'y_value', data= plot_data_technical, color=group_2_color(study_group), alpha=0.1, s=10, edgecolors=None)
    plt.scatter('mean_x', 'mean_y', data=plot_data_biological, color=group_2_color(study_group), alpha=0.4, s=50, marker='^', edgecolors=None)
    for joint in plot_data_summary['joint_name']:
        plt.scatter('mean_x', 'mean_y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.1, s=1000*plot_data_summary['std_norm'], marker='p')

    #Plot adjust
    sns.despine(left=True)
    plt.xlabel('Distance (x)', size=15, fontweight='bold')
    plt.ylabel('Distance (y)', size=15, fontweight='bold')
    plt.xticks(list(np.arange(-2, 4.5, 0.5)))
    plt.yticks(list(np.arange(0, 3.5, 0.5)))
    plt.title('Day Post SCI:'+' '+str(plot_day), size=15, fontweight='bold')
    plt.scatter(0.5, 3.1, s=100, color=group_2_color('sci'))
    plt.scatter(1, 3.1, s=100, color=group_2_color('sci_medium'))
    plt.scatter(1.8, 3.1, s=100, color=group_2_color('sci_msc'))
    plt.annotate('SCI', fontweight='bold', size=12, xy=(1, 1), xytext=(0.55, 3.075))
    plt.annotate('SCI+Medium', fontweight='bold', size=12, xy=(1, 1), xytext=(1.05, 3.075))
    plt.annotate('SCI+Medium+MSCs', fontweight='bold', size=12, xy=(1, 1), xytext=(1.85, 3.075))


def line_plotter(data_summary, plot_day, study_group, joint_comb):
    plot_data_summary = data_summary[(data_summary['day'] == plot_day) & (data_summary['group'] == study_group)&data_summary['joint_name'].isin(joint_comb)]

    plt.plot('mean_x', 'mean_y', data=plot_data_summary, color=group_2_color(study_group), alpha=0.7, linewidth=4)

#Removing origo before plotting
data_frame_boostrap_raw = data_frame_boostrap_raw[data_frame_boostrap_raw['joint_type']!='origo']
data_frame_boostrap_aggregate = data_frame_boostrap_aggregate[data_frame_boostrap_aggregate['joint_name']!='origo']
data_frame_boostrap_summary = data_frame_boostrap_summary[data_frame_boostrap_summary['joint_name']!='origo']

#Normalizing st.dev in summary data before plotting (to be able to adjust size of markers)
def std_normalizer(data_summary, plot_day):
    data_out = data_summary[data_summary['day']==plot_day]
    std_data = np.sqrt((data_out['std_x']**2+data_out['std_y']**2)/2)
    data_out = data_out.assign(std_norm=std_data/np.mean(std_data))
    return data_out

data_frame_boostrap_summary = pd.concat(list(map(lambda day: std_normalizer(data_frame_boostrap_summary, day),
                                                 list(data_frame_boostrap_summary['day'].unique()))), ignore_index=True)

def plot_caller(plot_day):
    list(map(lambda group: coordinates_overtime_plot_extended_v2(data_frame_boostrap_raw,
                                                                 data_frame_boostrap_aggregate,
                                                                 data_frame_boostrap_summary,
                                                                 plot_day, group), study_groups))

    list(map(lambda group: list(map(lambda joint_combo: line_plotter(data_frame_boostrap_summary, plot_day, group, joint_combo), joint_combinations)), study_groups))
    plt.savefig('position_plot_'+'d'+str(plot_day)+'.jpg', dpi=1000)

plot_caller(3)