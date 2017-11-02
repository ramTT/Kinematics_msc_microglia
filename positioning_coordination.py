import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#0. Defining color palette
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

#1. Importing data
data_side_coordinates_raw = pd.read_csv('merge_side_view.csv')
animal_key = pd.read_csv('animal_key_kinematics.csv')

#2. Selecting relevant columns
keep_columns = [': x [cm]', 'y [cm]', 'Joint 2: x [cm]', 'y [cm].1',
                'Joint 3: x [cm]', 'y [cm].2', 'Joint 4: x [cm]', 'y [cm].3', 'Joint 5: x [cm]', 'y [cm].4',
                'Joint 6: x [cm]', 'y [cm].5', 'RH.index', 'day', 'side']
data_side_coordinates = data_side_coordinates_raw[keep_columns]

#3. Renaming columns
new_column_names = {': x [cm]':'x_origo', 'y [cm]':'y_origo', 'Joint 2: x [cm]':'x_iliac', 'y [cm].1':'y_iliac',
                    'Joint 3: x [cm]':'x_trochanter', 'y [cm].2':'y_trochanter', 'Joint 4: x [cm]':'x_knee',
                    'y [cm].3':'y_knee', 'Joint 5: x [cm]':'x_ankle', 'y [cm].4':'y_ankle',
                    'Joint 6: x [cm]':'x_toe', 'y [cm].5':'y_toe'}

data_side_coordinates = data_side_coordinates.rename(columns = new_column_names)

#4. Adjusting RH.index
data_side_coordinates['RH.index'] = data_side_coordinates['RH.index'].apply(lambda index: index[:3])
data_side_coordinates['RH.index'] = data_side_coordinates['RH.index'].astype('int32')

#5. Merging with animal key
data_side_coordinates = data_side_coordinates.merge(animal_key[['RH.index', 'group']], on='RH.index')

#6. Normalizing x & y
def coordinate_normalizer(coord_type, data_frame):
    adjust_cols = [col_name for col_name in data_side_coordinates.columns if col_name[0] == coord_type]
    unadjusted_dict = dict(list(data_frame.groupby('side')))
    unadjusted_dict['left'] = unadjusted_dict['left'][adjust_cols].sub(unadjusted_dict['left'][coord_type+'_origo'], axis=0)
    unadjusted_dict['right'] = unadjusted_dict['right'][adjust_cols].sub(unadjusted_dict['right'][coord_type + '_origo'], axis=0)*(-1)

    adjusted_dict = pd.concat(unadjusted_dict, ignore_index=True)

    return adjusted_dict

data_side_coordinates_adjusted = pd.concat([coordinate_normalizer('x', data_side_coordinates).reset_index(drop=True),
           coordinate_normalizer('y', data_side_coordinates), data_side_coordinates[['RH.index', 'day', 'group']]], axis=1)

#7. Adjusting negative y-values in sci-group
y_columns = [col_name for col_name in data_side_coordinates_adjusted.columns if col_name[0]=='y']
data_side_coordinates_adjusted[y_columns] = data_side_coordinates_adjusted[y_columns].applymap(lambda value: value*(-1) if value<0 else value)

def melting_updating_casting(data_frame):
    out_data = data_frame.melt(id_vars=['RH.index', 'day', 'group'])

    out_data['coord_type'] = out_data['variable'].apply(lambda joint: joint[0])
    out_data['joint_name'] = out_data['variable'].apply(lambda joint: joint[2:])
    out_data = out_data.drop(['variable'], axis=1)

    out_data = out_data.pivot_table(index=['RH.index', 'day', 'group', 'joint_name'], values='value', columns='coord_type').reset_index()

    return out_data

data_side_coordinates_melt = melting_updating_casting(data_side_coordinates_adjusted)

#8. Adding force and displacement to dataset
data_side_coordinates_melt = data_side_coordinates_melt.merge(animal_key[['RH.index','force', 'displacement']], on='RH.index')

#9. Function for calc of displacement and force index
def indexator(data_frame, variable, day):
    data_frame[variable+'_index'] = data_frame[variable].apply(lambda disp: disp/min(data_frame[variable]))
    data_side_coordinates_melt.loc[day, variable+'_index'] = 1
    return data_frame

#10. Calling index function and
data_side_coordinates_melt = data_side_coordinates_melt.set_index('day')
data_side_coordinates_melt = list(map(lambda adjust_variable: indexator(data_side_coordinates_melt, adjust_variable, 3), ['displacement', 'force']))

#map() to dataframe?
#adjust x & y
#group adjustments
#hue on two parameters -> shape & color







#10. Plotting - biological replicates
side_overview_plot = sns.lmplot(data=data_side_coordinates_melt, x='x', y='y', fit_reg=False, col='day', hue='group',
           palette=palette_custom_1, legend=False)
side_overview_plot.set_xlabels('Distance [X]', size=10, fontweight='bold')
side_overview_plot.set_ylabels('Distance [Y]', size=10, fontweight='bold')
plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], ncol=3, loc='upper center')

#11. Plotting - average per group
test = data_side_coordinates_melt.groupby(['day', 'group', 'joint_name'], as_index=False)[['x', 'y']].mean()
ass = sns.FacetGrid(test, col='day', hue='group')
ass.map(sns.jointplot, 'x', 'y')