import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_import_adjust import data_side, data_side_aggregated
from data_import_adjust import data_bottom, data_bottom_aggregated
from data_import_adjust import data_angle, data_angle_aggregated

data_angle_subset = data_angle[['RH.index', 'day', 'angle_iliac_crest', 'group']]

data_angle_subset.loc[:,'angle_iliac_crest'] = data_angle_subset.loc[:,'angle_iliac_crest'].apply(lambda angle: (360-angle) if(angle>180) else angle)


