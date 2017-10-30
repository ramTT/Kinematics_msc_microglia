import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#1. Importing data
data_side_coordinates_raw = pd.read_csv('merge_side_view.csv')
animal_key = pd.read_csv('animal_key_kinematics.csv')
#2. Selecting relevant columns
keep_columns = [': x [cm]', 'y [cm]', 'Joint 2: x [cm]', 'y [cm].1',
                'Joint 3: x [cm]', 'y [cm].2', 'Joint 4: x [cm]', 'y [cm].3', 'Joint 5: x [cm]', 'y [cm].4',
                'Joint 6: x [cm]', 'y [cm].5', 'RH.index', 'day']
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
cols_ends = ['origo', 'iliac', 'trochanter', 'knee', 'ankle', 'toe']
x_cols = list(map(lambda end: 'x_'+end, cols_ends))
y_cols = list(map(lambda end: 'y_'+end, cols_ends))

data_side_coordinates[y_cols] = data_side_coordinates[y_cols].sub(data_side_coordinates['y_origo'], axis=0)
data_side_coordinates[x_cols] =  data_side_coordinates[x_cols].sub(data_side_coordinates['x_origo'], axis=0)

#7. Plot: over time, color by group
    #raw data
    #average data
    #draw line between averages?

#8. Calculate st.dev around points -> estimate coordination



