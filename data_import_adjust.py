import pandas as pd

def data_subset_rename(raw_data_set, select_columns, select_column_names):
    """Selects relevant columns and renames appropriately"""
    clean_data = raw_data_set.loc[:,select_columns]
    clean_data = clean_data.rename(columns=select_column_names)

    return clean_data


def add_group(data_set_main, data_set_key):
    """Merging main dataset with animal key"""
    if (isinstance(data_set_main['RH.index'], int)==True):
        data_set_merge = data_set_main.merge(data_set_key, on="RH.index")

    else:
        data_set_main['RH.index'] = data_set_main['RH.index'].apply(lambda index: index[:3]).astype('int64')
        data_set_merge = data_set_main.merge(data_set_key, on="RH.index")

    return data_set_merge


def data_aggregation_adjuster(data_not_aggregated):
    """Adjusts data after split-apply-combine calculation"""

    dataframe_new = pd.DataFrame(data_not_aggregated.values)

    col_names = []
    for variable in data_not_aggregated.columns.levels[0]:
        for desc_stat in data_not_aggregated.columns.levels[1]:
            col_names.append(variable + '_' + desc_stat)

    dataframe_new.columns = col_names

    counter = 0
    number_of_indices = len(data_not_aggregated.index.names)
    while counter < number_of_indices:
        dataframe_new[data_not_aggregated.index.names[counter]] = data_not_aggregated.index.get_level_values(level=counter)
        counter +=1

    return dataframe_new

#Importing data
data_side_view_raw = pd.read_csv('merge_side_view.csv')
data_bottom_view_raw = pd.read_csv('merge_bottom_view.csv')
data_key = pd.read_csv('animal_key_kinematics.csv')

#Defining relevant columns
side_view_columns = ['RH.index','side','day', 'Dist [cm].1', 'Dist [cm].2', 'Dist [cm].3', 'Dist [cm].4', 'Dist [cm].5']
bottom_view_columns = ['Dist [cm].1', 'RH.index', 'day']
angle_columns = ['RH.index', 'side', 'day','Ang [deg].1', 'Ang [deg].2', 'Ang [deg].3', 'Ang [deg].4']

#Defining new column names
side_view_column_names = {'Dist [cm].1':'iliac_crest_height', 'Dist [cm].2':'iliac_crest_2_trochanter_major', 'Dist [cm].3':'trochanter_major_2_knee',
                                           'Dist [cm].4':'knee_2_ankle', 'Dist [cm].5':'ankle_2_toe'}
bottom_view_column_names = {'Dist [cm].1': 'inter_knee_distance'}
angle_column_names = {'Ang [deg].1':'angle_iliac_crest', 'Ang [deg].2':'angle_trochanter_major', 'Ang [deg].3':'angle_knee',
                                        'Ang [deg].4':'angle_ankle'}

#Subsetting raw data and renaming columns
data_side = data_subset_rename(data_side_view_raw, side_view_columns, side_view_column_names)
data_bottom = data_subset_rename(data_bottom_view_raw, bottom_view_columns, bottom_view_column_names)
data_angle = data_subset_rename(data_side_view_raw, angle_columns, angle_column_names)

#Merging datasets with data_key
data_side = add_group(data_side, data_key)
data_bottom = add_group(data_bottom, data_key)
data_angle = add_group(data_angle, data_key)

#4. Aggregate data per animal
data_side_aggregated = data_aggregation_adjuster(data_side.groupby(['day', 'group']).agg({'iliac_crest_height':['mean']}))
data_angle_aggregated = data_aggregation_adjuster(data_angle.groupby(['RH.index', 'day']).agg({'angle_iliac_crest':['mean'], 'angle_trochanter_major': ['mean'], 'angle_knee': ['mean']}))
data_bottom_aggregated = data_aggregation_adjuster(data_bottom.groupby(['day', 'group']).agg({'inter_knee_distance':['mean']}))