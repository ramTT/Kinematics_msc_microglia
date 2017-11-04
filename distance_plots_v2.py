import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DistancePlot:

    def __init__(self, data_frame):
        self.data_frame = data_frame

    def column_adjuster(self, keep_cols, new_column_names):
        self.data_frame = self.data_frame[keep_cols]
        self.data_frame = self.data_frame.rename(columns = new_column_names)

    def key_data_adder(self, key_data):
        self.data_frame['RH.index'] = self.data_frame['RH.index'].map(lambda rh_index: rh_index[:3])
        self.data_frame['RH.index'] = self.data_frame['RH.index'].astype('int64')
        self.data_frame = self.data_frame.merge(key_data, on='RH.index')

    def measure_adjuster(self, adjust_variable, measure='displacement'):
        self.data_frame['adjustor_column'] = self.data_frame[measure].map(lambda value: value/min(self.data_frame[measure]))
        self.data_frame.loc[self.data_frame['day']==3, 'adjustor_column'] = 1
        self.data_frame[adjust_variable+'_adjust'] = self.data_frame[adjust_variable]*self.data_frame['adjustor_column']

    def indexator(self, variable):
        dictionary = dict(list(self.data_frame.groupby('RH.index')))
        keys = dictionary.keys()

        def normalizer(dict, dict_key):
            #divisor = dict[dict_key].loc[dict[dict_key]['day'] == 3, variable].mean()
            divisor = self.data_frame.min()[variable]
            dict[dict_key][variable] = dict[dict_key][variable] / divisor
            return dict[dict_key]

        self.data_frame = pd.concat(list(map(lambda key: normalizer(dictionary, key), keys)), axis=0)

    def column_cleanup(self, keep_columns):
        self.data_frame = self.data_frame[keep_columns]

    def aggregate_per_animal(self):
        return self.data_frame.groupby(['RH.index', 'day', 'group'], as_index=False).median()

    def summary(self):
        return self.data_frame.groupby(['day', 'group'], as_index=False).mean()

#0. Importing key data (RH.index, group, force and displacment)
animal_key = pd.read_csv('animal_key_kinematics.csv')

#1. Managing ILIAC CREST HEIGHT
#A. Importing data, creating variables
data_set_iliac = pd.read_csv('merge_side_view.csv')

keep_columns_iliac = ['RH.index', 'day', 'Dist [cm].1', 'Dist [cm].2', 'Dist [cm].3', 'Dist [cm].4', 'Dist [cm].5']
new_column_names_iliac = {'Dist [cm].1':'iliac_crest_height', 'Dist [cm].2':'iliac_crest_2_trochanter_major', 'Dist [cm].3':'trochanter_major_2_knee',
                                           'Dist [cm].4':'knee_2_ankle', 'Dist [cm].5':'ankle_2_toe'}
#B. Creating instance, calling methods
instance_iliac = DistancePlot(data_set_iliac)
instance_iliac.column_adjuster(keep_columns_iliac, new_column_names_iliac)
instance_iliac.key_data_adder(animal_key)
instance_iliac.measure_adjuster('iliac_crest_height')

instance_iliac.indexator('iliac_crest_height_adjust')
instance_iliac.column_cleanup(['RH.index', 'day', 'group', 'iliac_crest_height_adjust'])


data_aggregate_iliac = instance_iliac.aggregate_per_animal()
data_summary_iliac = instance_iliac.summary()

#2. Managing INTER KNEE DISTANCE
#A. Importing data, creating variables
data_set_knee = pd.read_csv('merge_bottom_view.csv')

keep_columns_knee = ['Dist [cm].1', 'RH.index', 'day']
new_column_names_knee = {'Dist [cm].1': 'inter_knee_distance'}

#B. Creating instance, calling methods
instance_knee = DistancePlot(data_set_knee)
instance_knee.column_adjuster(keep_columns_knee, new_column_names_knee)
instance_knee.key_data_adder(animal_key)
instance_knee.measure_adjuster('inter_knee_distance')
instance_knee.indexator('inter_knee_distance_adjust')
instance_knee.column_cleanup(['RH.index', 'day', 'group', 'inter_knee_distance_adjust'])

data_aggregate_knee = instance_knee.aggregate_per_animal()
data_summary_knee = instance_knee.summary()

#3. Defining color palette for plotting
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

#4. Plotting
def over_time_plot(data_set, y_variable, y_label):
    #sns.stripplot(data = data_set, x='day', y=y_variable, hue='group', size=15, palette=palette_custom_1,
    #              dodge=True, jitter=0.25, alpha=0.5, marker='o')
    sns.regplot(data = data_set, x='day', y=y_variable, hue='group')
    sns.despine(left=True, top=True)
    plt.xlabel('Day (Post SCI)', size=20, fontweight='bold')
    plt.ylabel(y_label, size=20, fontweight='bold')
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], loc='lower center', frameon=False, ncol=3, fontsize=12)

over_time_plot(instance_iliac.data_frame, 'iliac_crest_height_adjust', 'Iliac crest height index')

#regplot
    #1 for each group -> raw data
    #2 for each group -> biological replicates
    #1 lowess for data




