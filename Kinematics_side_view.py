#Attaching packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################################################################ FUNCTIONS ######################################################################

def post_agg_dataframe_constructor(ingoing_dataframe_with_multiindex):
    number_of_indices = len(ingoing_dataframe_with_multiindex.index.names)

    dataframe_new = pd.DataFrame(ingoing_dataframe_with_multiindex.values)
    dataframe_new.columns = ingoing_dataframe_with_multiindex.columns.levels[1]

    counter = 0
    while counter < number_of_indices:
        dataframe_new[ingoing_dataframe_with_multiindex.index.names[counter]] = ingoing_dataframe_with_multiindex.index.get_level_values(level=counter)
        counter +=1

    return dataframe_new

################################################################ IMPORTING DATA & CREATING DATA SUBSETS ######################################################################
#1. Importing data
DT = pd.read_csv("merge_side_view.csv")

#A. DISTANCE DATA SUBSET
DT_distance = DT.loc[:,['RH.index','side','day', 'Dist [cm].1', 'Dist [cm].2', 'Dist [cm].3', 'Dist [cm].4', 'Dist [cm].5']]
# Renaming columns
DT_distance = DT_distance.rename(columns ={'Dist [cm].1':'iliac_crest_height', 'Dist [cm].2':'iliac_crest_2_trochanter_major', 'Dist [cm].3':'trochanter_major_2_knee',
                                           'Dist [cm].4':'knee_2_ankle', 'Dist [cm].5':'ankle_2_toe'})

#Adjusting RH_index (removing additions)
DT_distance['RH.index'] = DT_distance['RH.index'].apply(lambda x: x[:3])
DT_distance['RH.index'] = DT_distance['RH.index'].astype('int64')

#B. ANGLE DATA SUBSET
DT_angles = DT.loc[:,['RH.index', 'side', 'day','Ang [deg].1', 'Ang [deg].2', 'Ang [deg].3', 'Ang [deg].4']]
#Renaming columns
DT_angles = DT_angles.rename(columns = {'Ang [deg].1':'angle_iliac_crest', 'Ang [deg].2':'angle_trochanter_major', 'Ang [deg].3':'angle_knee',
                                        'Ang [deg].4':'angle_ankle'})

#Adjusting iliac_crest and trochanter major angles to 360-angle
DT_angles[['angle_iliac_crest', 'angle_trochanter_major']] = DT_angles[['angle_iliac_crest', 'angle_trochanter_major']].apply(lambda x: 360-x, axis=0)

#Adjusting RH.index (removing additions)
DT_angles['RH.index'] = DT['RH.index'].apply(lambda x: x[:3])
DT_angles['RH.index'] = DT_angles['RH.index'].astype('int64')

#C. ADDING STUDY GROUP BASED ON RH INDEX
animal_key = pd.read_csv("animal_key_kinematics.csv")
category_vars = ['group']
animal_key[category_vars] = animal_key[category_vars].apply(lambda x: x.astype('category'), axis=0)

DT_distance = DT_distance.merge(animal_key, left_on='RH.index', right_on='RH.index')
DT_angles = DT_angles.merge(animal_key, on='RH.index')

################################################################  AGGREGATING DATA ######################################################################

#1. AGGREGATING ON INDIVIDUAL LEVEL (FOR STATISTICAL ANALYSIS & PLOTTING)

#A. DISTANCE

DT_distance_aggregate_raw = DT_distance.groupby(['RH.index', 'day']).agg({'iliac_crest_height' : {'mean'}})
#Adjusting output with post_agg adjuster function
DT_distance_aggregate = post_agg_dataframe_constructor(DT_distance_aggregate_raw)
#Merging aggregate dataset with animal_key
DT_distance_aggregate.merge(animal_key, on="RH.index")

#B. Angles

DT_angles


#2. AGGREGATING ON TREATMENT GROUP LEVEL(FOR SUMMARY & PLOTTING)

DT_distance['iliac_crest_height'].head()



################################################################  STATISTICAL ANALYSIS ######################################################################

################################################################  PLOTTING DATA ######################################################################



