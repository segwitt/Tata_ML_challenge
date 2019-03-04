# to visualize the dataset


import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd

import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



random_seed = 42
random.seed(random_seed)

'''
do not change this code
the snippet generates the datasets for modelling
'''

# reading the data from the given raw file
raw_dat = pd.read_csv('DSDataLastThreeMonths.csv')
# data2 = raw_dat.copy()
# removing null values for further analysis
raw_dat = raw_dat.dropna().reset_index()
data2 = raw_dat.copy()


# mean = data2.loc[:,'HM_WT':'MG_INJ_TIME'].mean()
# print(mean)
# data2.loc[:,'HM_WT':'MG_INJ_TIME'] -= mean
# sigma = data2.loc[:,'HM_WT':'MG_INJ_TIME'].pow(2).mean()
# sigma = sigma.pow(0.5)
# print(sigma)
# data2.loc[:,'HM_WT':'MG_INJ_TIME']/=sigma
# # print(data2.loc[:,'HM_WT':'MG_INJ_TIME'])



# raw_dat = data2


# extracting the independent variable, X
x_data = raw_dat.loc[:,['HM_WT', 'AIM_S', 'HM_S', 'HM_C', 'HM_SI', 'HM_TI','HM_MN', 'CAC2', 'MG', 'HM_TEMP', 'CAC2_INJ_TIME', 'MG_INJ_TIME']]
# extracting the dependent variable, y
y_data = raw_dat.loc[:,'DS_S']

# print(raw_dat.loc[:,'CAC2'])
print(raw_dat.loc[:,'CAC2_INJ_TIME']/raw_dat.loc[:,'MG_INJ_TIME'])


# splitting the entire dataset into test and train
# for all model development, use only the train dataset
# do not touch the test dataset for any development purposes
# test dataset is purely for validation purposes
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)






import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import math
# plt.plot(X_train['AIM_S'])
# plt.plot(list(X_train.iloc[300:400, 5]), label='in5')
# plt.plot(list(X_train.iloc[300:400, 4]), label='in4')#same as prev
# plt.plot(list(X_train.iloc[300:400, 0]), label='in0')#opp of prev
# plt.plot(list(X_train.iloc[300:400, 1]), label='in1')
# plt.plot(list(X_train.iloc[300:400, 2]), label='in2')
# plt.plot(list(X_train.iloc[300:400, 3]), label='in3')
# plt.plot(list(X_train.iloc[300:400, 6]), label='in6')
# plt.plot(list(X_train.iloc[300:400, 7]), label='in7')
# plt.plot(list(X_train.iloc[300:400, 8]), label='in8') # almost same as prev
# plt.plot(list(X_train.iloc[300:400, 9]), label='in9')
# plt.plot(list(X_train.iloc[300:400, 10]), label='in10')
# plt.plot(list(X_train.iloc[300:400, 11]), label='in11')
# plt.plot(list(y_train.iloc[300:400]), label='out')
# plt.legend(frameon=False)
# plt.show()
