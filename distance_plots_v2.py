import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1. Importing data
data_set_raw = pd.read_csv('merge_side_view.csv')
animal_key = pd.read_csv('animal_key_kinematics.csv')

#2. Removing unnecessary columns
keep_columns = ['RH.index', 'day', 'Dist [cm].1', 'Dist [cm].2', 'Dist [cm].3', 'Dist [cm].4', 'Dist [cm].5']
data_set = data_set_raw[keep_columns]

#3. Renaming columns
new_column_names = {'Dist [cm].1':'iliac_crest_height', 'Dist [cm].2':'iliac_crest_2_trochanter_major', 'Dist [cm].3':'trochanter_major_2_knee',
                                           'Dist [cm].4':'knee_2_ankle', 'Dist [cm].5':'ankle_2_toe'}

data_set = data_set.rename(columns = new_column_names)

#4. Merging with animal key
data_set['RH.index'] = data_set['RH.index'].map(lambda rh_index: rh_index[:3])
data_set['RH.index'] = data_set['RH.index'].astype('int64')
data_set = data_set.merge(animal_key, on='RH.index')

#5. Keeping only iliac crest height
data_set_subset = data_set[['RH.index', 'day', 'group', 'displacement', 'iliac_crest_height']]

#6. Creating displacement index
data_set_subset = data_set_subset.assign(displacement_index = lambda disp: disp.displacement/min(disp.displacement))

#7.Adjusting iliac crest height with displacement index
data_set_subset = data_set_subset.assign(iliac_crest_height_adjust = data_set_subset['iliac_crest_height']*data_set_subset['displacement_index'])

#8. Aggregating per animal (RH.index)
data_set_subset_aggregate = data_set_subset.groupby(['RH.index', 'day', 'group'], as_index=False).mean()

#9. Aggregating per group
data_set_subset_summary = data_set_subset.groupby(['day', 'group'], as_index=False).mean()

#10. Defining color palette
palette_BrBG = pd.DataFrame(list(sns.color_palette("BrBG", 7)))
palette_RdBu_r = pd.DataFrame(list(sns.color_palette("RdBu_r", 7)))
palette_custom_1 = [tuple(palette_BrBG.iloc[0,:]), tuple(palette_RdBu_r.iloc[0,:]), tuple(palette_RdBu_r.iloc[6,:])]

#11. Plotting
sns.stripplot(data = data_set_subset, x='day', y='iliac_crest_height_adjust', hue='group', size=5, palette=palette_custom_1,
              dodge=True, jitter=0.25, alpha=0.5, marker='*')
sns.despine(left=True, top=True)
plt.xlabel('Day (Post SCI)', size=12)
plt.ylabel('Iliac crest height', size=12)
plt.title('Iliac crest height', size=15, fontweight='bold')
plt.legend(['SCI', 'SCI+Medium', 'SCI+Medium+IDmBMSCs'], loc='lower center', frameon=False, ncol=3, fontsize=8)




