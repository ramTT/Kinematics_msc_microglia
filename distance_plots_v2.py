import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class KinematicsDataAdjuster:

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

    def indexator(self, variable, denominator):
        dictionary = dict(list(self.data_frame.groupby('RH.index')))
        keys = dictionary.keys()

        def normalizer(dict, dict_key):
            if(denominator=='day3mean'):
                divisor = dict[dict_key].loc[dict[dict_key]['day'] == 3, variable].mean()
            elif(denominator=='day3min'):
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
data_set_iliac = pd.read_csv('merge_side_view.csv', encoding='utf-16')

keep_columns_iliac = ['RH.index', 'day', 'Dist [cm].1', 'Dist [cm].2', 'Dist [cm].3', 'Dist [cm].4', 'Dist [cm].5']
new_column_names_iliac = {'Dist [cm].1':'iliac_crest_height', 'Dist [cm].2':'iliac_crest_2_trochanter_major', 'Dist [cm].3':'trochanter_major_2_knee',
                                           'Dist [cm].4':'knee_2_ankle', 'Dist [cm].5':'ankle_2_toe'}
#B. Creating instance, calling methods
instance_iliac = KinematicsDataAdjuster(data_set_iliac)
instance_iliac.column_adjuster(keep_columns_iliac, new_column_names_iliac)
instance_iliac.key_data_adder(animal_key)
instance_iliac.measure_adjuster('iliac_crest_height')

instance_iliac.indexator('iliac_crest_height_adjust', 'day3min')
instance_iliac.column_cleanup(['RH.index', 'day', 'group', 'iliac_crest_height_adjust'])

data_aggregate_iliac = instance_iliac.aggregate_per_animal()
data_summary_iliac = instance_iliac.summary()

#2. Managing INTER KNEE DISTANCE
#A. Importing data, creating variables
data_set_knee = pd.read_csv('merge_bottom_view.csv')

keep_columns_knee = ['Dist [cm].1', 'RH.index', 'day']
new_column_names_knee = {'Dist [cm].1': 'inter_knee_distance'}

#B. Creating instance, calling methods
instance_knee = KinematicsDataAdjuster(data_set_knee)
instance_knee.column_adjuster(keep_columns_knee, new_column_names_knee)
instance_knee.key_data_adder(animal_key)
instance_knee.measure_adjuster('inter_knee_distance')
instance_knee.data_frame['inter_knee_distance_adjust'] = instance_knee.data_frame['inter_knee_distance']/instance_knee.data_frame['adjustor_column']
instance_knee.indexator('inter_knee_distance_adjust', 'day3mean')
instance_knee.column_cleanup(['RH.index', 'day', 'group', 'inter_knee_distance_adjust'])

data_aggregate_knee = instance_knee.aggregate_per_animal()
data_summary_knee = instance_knee.summary()

#3. Defining color palette for plotting
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

def group_2_color(argument):
    '''Dictionary mapping (replacing switch/case statement)'''
    switcher = {
        'sci': palette_custom_1[0],
        'sci_medium': palette_custom_1[1],
        'sci_msc': palette_custom_1[2]
    }
    return switcher.get(argument)

study_groups = ['sci', 'sci_medium', 'sci_msc']

#4. Plotting distributions
def distribution_plot(plot_data, x_var, x_label, color_palette, x_limits):
    out_plot = sns.FacetGrid(plot_data, row='day', hue='group', aspect=3, size=3, palette=color_palette)
    out_plot.map(sns.kdeplot, x_var, clip_on=False, shade=True, alpha=0.5, lw=1.5, bw=0.2, kernel='cos')
    out_plot.map(sns.kdeplot, x_var, clip_on=False, color='w', lw=2, bw=0.2, kernel='cos')

    for row, day in enumerate(['3', '7', '14', '21', '28', '35', '42']):
        out_plot.axes[row, 0].set_ylabel('Day ' + day, size=15, fontweight='bold')

    out_plot.set_titles('')
    out_plot.set(yticks=[])
    out_plot.despine(left=True)
    plt.xlabel(x_label, size=20, fontweight='bold')
    plt.xticks(list(np.arange(0.5, 5, 0.25)))
    plt.xlim(x_limits)

distribution_plot(data_aggregate_iliac, 'iliac_crest_height_adjust', 'Iliac crest height index', palette_custom_1, [0.7, 3.25])
#plt.savefig('distribution_plot_iliac.jpg', dpi=1000)
distribution_plot(data_aggregate_knee, 'inter_knee_distance_adjust', 'Inter knee distance index', palette_custom_1, [0.25,2.75])
#plt.savefig('distribution_plot_knee.jpg', dpi=1000)

#5. Plotting over time
def day_adjuster(dataset):
    dataset.loc[dataset['group'] == 'sci', ['day']] += 1
    dataset.loc[dataset['group'] == 'sci_medium', ['day']] -= 1

datasets_iliac = [instance_iliac.data_frame, data_aggregate_iliac, data_summary_iliac]
datasets_knee = [instance_knee.data_frame, data_aggregate_knee, data_summary_knee]

[day_adjuster(data_set) for data_set in datasets_iliac]
[day_adjuster(data_set) for data_set in datasets_knee]

def overtime_plot(data_technical, data_biological, data_summary, study_group, y_variable, y_title):
    #Creating plot data
    plot_data_technical = data_technical[data_technical['group']==study_group]
    plot_data_biological = data_biological[data_biological['group']==study_group]
    plot_data_summary = data_summary[data_summary['group']==study_group]

    #Creating plots
    plt.scatter('day', y_variable, data = plot_data_technical, color = group_2_color(study_group), s=50, alpha=0.1)
    plt.scatter('day', y_variable, data = plot_data_biological, color = group_2_color(study_group), s=300,
                alpha=0.4, marker="^")

    plt.scatter('day', y_variable, data=plot_data_summary, color=group_2_color(study_group), s=1000,
                alpha=0.8, marker="p")

    plt.plot('day', y_variable, data=plot_data_summary, color=group_2_color(study_group),alpha=0.8, lw=5)

    #Plot adjust
    sns.despine(left=True)
    plt.xlabel('Day (Post SCI)', size=15, fontweight='bold')
    plt.ylabel(y_title, size=15, fontweight='bold')
    plt.xticks(list(np.arange(0, 49, 7)))
    plt.yticks(list(np.arange(0.75, 3.5, 0.5)))
    plt.scatter(8,3.5, s=100, color=group_2_color('sci'))
    plt.scatter(18, 3.5, s=100, color=group_2_color('sci_medium'))
    plt.scatter(28, 3.5, s=100, color=group_2_color('sci_msc'))
    plt.annotate('SCI', xy=(1,1), xytext=(9, 3.475), fontweight='bold', size=12)
    plt.annotate('SCI+Medium', xy=(1, 1), xytext=(19, 3.475), fontweight='bold', size=12)
    plt.annotate('SCI+Medium+MSCs', xy=(1, 1), xytext=(29, 3.475), fontweight='bold', size=12)

list(map(lambda group: overtime_plot(instance_iliac.data_frame, data_aggregate_iliac,data_summary_iliac,group, 'iliac_crest_height_adjust', 'Iliac crest height index'), study_groups))
#plt.savefig('iliac_crest_height_overtime.jpg', dpi=1000)
list(map(lambda group: overtime_plot(instance_knee.data_frame, data_aggregate_knee,data_summary_knee, group, 'inter_knee_distance_adjust', 'Inter knee distance'), study_groups))
#plt.savefig('inter_knee_distance_overtime.jpg', dpi=1000)