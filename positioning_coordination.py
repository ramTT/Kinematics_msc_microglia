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
