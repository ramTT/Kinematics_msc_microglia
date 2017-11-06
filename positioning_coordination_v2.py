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
        del data_melt['variable']

        self.data_frame_melt = data_melt.pivot_table(index=['RH.index', 'day', 'group', 'joint_name'],
                                         values='value', columns='coord_type').reset_index()

    def adjustor(self, adjust_var):
        self.data_frame_melt[adjust_var + '_adjust'] = self.data_frame_melt[adjust_var] * \
                                                                    self.data_frame_melt['displacement']

#1. Importing data
data_set_position = pd.read_csv('merge_side_view.csv')
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

#D. Melting data
instance_position.melting_updating_casting()

#E. Adding force and displacement to aggregated dataset
instance_position.data_frame_melt = instance_position.key_data_adder(animal_key.drop(['group'], axis=1), instance_position.data_frame_melt)

#F. Normalizing displacement and removing force
del instance_position.data_frame_melt['force']
instance_position.data_frame_melt['displacement'] = instance_position.data_frame_melt['displacement'].map(lambda value:
                                                      value / min(instance_position.data_frame_melt['displacement']))

instance_position.data_frame_melt.loc[instance_position.data_frame_melt['day']==3, ['displacement']] = 1

#G. Adjusting x -and y coordinates using displacement
[instance_position.adjustor(element) for element in ['x', 'y']]







#0. Defining color palette
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

#10. Plotting - biological replicates
side_overview_plot = sns.lmplot(data = data_side_coordinates_melt, x='displacement_index_x', y='displacement_index_y', hue='group',
                                   legend=False, col='day', fit_reg=False, palette=palette_custom_1)
side_overview_plot.set_xlabels('Distance [X]', size=10, fontweight='bold')
side_overview_plot.set_ylabels('Distance [Y]', size=10, fontweight='bold')

#11. Plotting - average per group
average_data = data_side_coordinates_melt.groupby(['day', 'group', 'joint_name'], as_index=False).mean()
average_data['RH.index'] = average_data['RH.index'].astype('object')
sns.lmplot(data = average_data, x='displacement_index_x', y='displacement_index_y', col='day', hue='group',
           fit_reg=False, palette=palette_custom_1)