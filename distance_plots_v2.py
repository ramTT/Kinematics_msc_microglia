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
            divisor = dict[dict_key].loc[dict[dict_key]['day'] == 3, variable].mean()
            #divisor = self.data_frame.min()[variable]
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

#4. Plotting raw data over time
def plot_raw_data_overtime(data_frame, y_variable, y_label):
    sns.stripplot(data = data_frame, x='day', y= y_variable, hue='group', dodge=True,palette=palette_custom_1,
                  jitter=0.3, alpha=0.7)
    sns.despine(left=True)
    plt.xlabel('Day (Post SCI)', size=15, fontweight='bold')
    plt.ylabel(y_label, size=15, fontweight='bold')
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], frameon=False, ncol=3, loc='lower center')
    plt.yticks(list(np.arange(0.75,3.5,0.25)))

plot_raw_data_overtime(instance_iliac.data_frame, 'iliac_crest_height_adjust', 'Iliac crest height')
#plt.savefig('iliac_crest_raw_overtime.jpg', dpi=1000)
plot_raw_data_overtime(instance_knee.data_frame, 'inter_knee_distance_adjust', 'Inter knee distance')
#plt.savefig('inter_knee_raw_overtime.jpg', dpi=1000)

#5. Plotting aggregate data over time
def plot_aggregate_data_overtime(data_frame, y_variable, y_label):
    sns.lmplot(data=data_frame, x='day', y=y_variable, hue='group', palette=palette_custom_1,
                  x_jitter=1, lowess=True, legend_out=False, scatter_kws={'alpha':0.7, 's':200}, markers=['p','o','^'],
               size=5, aspect=2)
    sns.despine(left=True)
    plt.xlabel('Day (Post SCI)', size=12, fontweight='bold')
    plt.ylabel(y_label, size=12, fontweight='bold')
    plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], frameon=False, ncol=3, loc='lower center')
    plt.yticks(list(np.arange(1, 3, 0.25)))
    plt.xticks(list(np.arange(0,35,7)))

plot_aggregate_data_overtime(data_aggregate_iliac, 'iliac_crest_height_adjust', 'Iliac crest height index')
#plt.savefig('iliac_crest_aggregate_overtime.jpg', dpi=1000)
plot_aggregate_data_overtime(data_aggregate_knee, 'inter_knee_distance_adjust', 'Inter knee distance index')
#plt.savefig('inter_knee_aggregate_overtime.jpg', dpi=1000)

#6. Plotting correlations
#A. Creating dictionaries
dictionary_iliac = dict(list(instance_iliac.data_frame.groupby(['RH.index', 'day'])))
dictionary_knee = dict(list(instance_knee.data_frame.groupby(['RH.index', 'day'])))
#B. Boostraping from each RH.index and day (n=10)
corr_data_iliac = pd.concat(list(map(lambda key: dictionary_iliac[key].sample(n=10), list(dictionary_knee.keys()))), axis=0, ignore_index=True)
corr_data_knee = pd.concat(list(map(lambda key: dictionary_knee[key].sample(n=10), list(dictionary_knee.keys()))), axis=0, ignore_index=True)
#C. Binding data by columns
corr_plot_data = pd.concat([corr_data_iliac, corr_data_knee.drop(['RH.index', 'day', 'group'], axis=1)], axis=1)

