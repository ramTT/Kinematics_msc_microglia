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
data_side_coordinates = data_side_coordinates.merge(animal_key, on='RH.index')

#6. Reseting x_base and y_base to 0
def reset_base_origo(coordinate_type, data_frame):
    selected_cols = filter(lambda col_name: (col_name[0]==coordinate_type) | (col_name=='side'), data_frame.columns)
    selected_cols = list(selected_cols)
    group_on_side_dict = dict(list((data_frame[selected_cols].groupby(['side']))))

    def origo_adjuster(dictionary, chapter):
        data = dictionary[chapter].drop('side', axis=1)
        if(chapter=='left'):
            data = data.sub(data[coordinate_type+'_origo'], axis=0)
            return data
        elif(chapter=='right'):
            data = data.sub(data[coordinate_type+'_origo'], axis=0)*(-1)
            return data



    return pd.concat([origo_adjuster(group_on_side_dict,'left'), origo_adjuster(group_on_side_dict,'right')], ignore_index=True)

test_x = reset_base_origo('x', data_side_coordinates)
test_y = reset_base_origo('y', data_side_coordinates)
ass = pd.concat([test_x.reset_index(drop=True), test_y, data_side_coordinates[['RH.index', 'day', 'group']]], axis=1)
ass = ass.melt(id_vars=['RH.index', 'day', 'group'])
















#Plotting

#1. Justera y -> ta hänsyn till om left eller right side
#2. Bättre melt metod
#3. Kombinera alla mått i en plot för bättre överblick. Med lineplot för att visa mönstret (minus iliac crest height)

data_side_coordinates.head(2)
#Iliac crest height
sns.lmplot('x_iliac', 'y_iliac', data=data_side_coordinates, col='day', fit_reg=False,
                       hue='group', palette=palette_custom_1, markers='*')
#Trochanter major
sns.lmplot('x_trochanter', 'y_trochanter', data=data_side_coordinates, col='day', fit_reg=False,
                       hue='group', palette=palette_custom_1, markers='+')
#Knee
sns.lmplot('x_knee', 'y_knee', data=data_side_coordinates, col='day', fit_reg=False,
                       hue='group', palette=palette_custom_1, markers='+')
#Ankle
sns.lmplot('x_ankle', 'y_ankle', data=data_side_coordinates, col='day', fit_reg=False,
                       hue='group', palette=palette_custom_1, markers='+')

#Toe
sns.lmplot('x_toe', 'y_toe', data=data_side_coordinates, col='day', fit_reg=False,
                       hue='group', palette=palette_custom_1, markers='+')









#7. Melting dataset
def melt_dataset(selected_columns):
    extension_variables = ['RH.index', 'day', 'group']
    selected_columns.extend(extension_variables)

    melted_data = data_side_coordinates[selected_columns].melt(id_vars=extension_variables)
    melted_data['variable'] = melted_data['variable'].apply(lambda coord_type: coord_type[2:])
    return melted_data

data_side_coordinates_long = melt_dataset(reset_base_origo('x')).merge(melt_dataset(reset_base_origo('y')), on=['RH.index', 'day', 'group'])
data_side_coordinates_long = data_side_coordinates_long.drop('variable_y', axis=1)
data_side_coordinates_long = data_side_coordinates_long.rename(columns={'variable_x':'variable'})

#7. Plot: over time, color by group
    #raw data, #average data, #draw line between averages?, #Development per group over time