#Attaching packages
import pandas as pd
import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns

################################################################ FUNCTIONS ######################################################################
def post_agg_dataframe_constructor(ingoing_dataframe_with_multiindex):
    """Function adjusts dataframe after split-apply combine operation on dataframe"""

    #Creating pd.DataFrame of ingoing data (output from split-apply-combine)
    dataframe_new = pd.DataFrame(ingoing_dataframe_with_multiindex.values)

    #Creating column names (variable + descriptive_statistic)
    col_names = []
    for variable in ingoing_dataframe_with_multiindex.columns.levels[0]:
        for desc_stat in ingoing_dataframe_with_multiindex.columns.levels[1]:
            col_names.append(variable + '_' + desc_stat)

    dataframe_new.columns = col_names

    #Adding groping variables to the new DataFrame
    counter = 0
    number_of_indices = len(ingoing_dataframe_with_multiindex.index.names)
    while counter < number_of_indices:
        dataframe_new[ingoing_dataframe_with_multiindex.index.names[counter]] = ingoing_dataframe_with_multiindex.index.get_level_values(level=counter)
        counter +=1

    return dataframe_new


def angle_adjust_function(angle):
    if (angle>180):
        return 360-angle
    elif (angle<180):
        return angle

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
DT_angles.iloc[:,3:7] = DT_angles.iloc[:,3:7].applymap(angle_adjust_function)

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
DT_distance_aggregate_raw = DT_distance.groupby(['RH.index', 'day']).agg({'iliac_crest_height' : ['mean']})
DT_distance_aggregate = post_agg_dataframe_constructor(DT_distance_aggregate_raw)

#Adjusting output with post_agg adjuster function
DT_distance_aggregate = post_agg_dataframe_constructor(DT_distance_aggregate_raw)

#Merging aggregate dataset with animal_key
DT_distance_aggregate = DT_distance_aggregate.merge(animal_key, on="RH.index")

#B. Angles
aggregate_variables = {'angle_iliac_crest':['mean'], 'angle_trochanter_major': ['mean'], 'angle_knee': ['mean']}
DT_angles_aggregate_raw = DT_angles.groupby(['RH.index', 'day']).agg(aggregate_variables)

#Adjusting output with post_agg adjuster function
DT_angles_aggregate = post_agg_dataframe_constructor(DT_angles_aggregate_raw)

#Merging dataset with animal_key
DT_angles_aggregate = DT_angles_aggregate.merge(animal_key, on="RH.index")

#2. AGGREGATING ON TREATMENT GROUP LEVEL (FOR SUMMARY & PLOTTING

#A. DISTANCE: Adding confidence intervals
DT_distance_summary = DT_distance_aggregate.groupby(['day', 'group']).agg({'iliac_crest_height_mean':['mean', 'std', 'count']})
DT_distance_summary = post_agg_dataframe_constructor(DT_distance_summary)
DT_distance_summary['SEMx1.96'] = 1.96*(DT_distance_summary['iliac_crest_height_mean_std'] / np.sqrt(DT_distance_summary['iliac_crest_height_mean_mean']))
DT_distance_summary['CI.lower'] = DT_distance_summary['iliac_crest_height_mean_mean']-DT_distance_summary['SEMx1.96']
DT_distance_summary['CI.Upper'] = DT_distance_summary['iliac_crest_height_mean_mean']+DT_distance_summary['SEMx1.96']

#B. ANGLES: Adding confidence intervals

################################################################  STATISTICAL ANALYSIS ######################################################################

################################################################  PLOTTING DATA ######################################################################
#Iliac crest height index over time
#angles over time -> widening should be good
#relation between height, angles and leg spread




