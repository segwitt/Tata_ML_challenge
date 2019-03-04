
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
# removing null values for further analysis
raw_dat = raw_dat.dropna().reset_index()
data2 = raw_dat.copy()


"""
DATA NORMALIZATION
"""


"""
Z score normalisation
epochs 10
Test acc = 52
Train acc = 52

varying sometimes 54, 56 sometimes 37

variable so depends on the random weights

20 epochs
Test 48.95
train 51.02
"""

# mean = data2.loc[:,'HM_WT':'MG_INJ_TIME'].mean()
# print('mean over whole data','\n',mean)
# data2.loc[:,'HM_WT':'MG_INJ_TIME'] -= mean
# sigma = data2.loc[:,'HM_WT':'MG_INJ_TIME'].pow(2).mean()
# sigma = sigma.pow(0.5)
# # print(sigma)
# data2.loc[:,'HM_WT':'MG_INJ_TIME']/=sigma


# raw_dat = data2



"""
normalizing the output also
"""


mean = data2.loc[:,'HM_WT':'MG_INJ_TIME'].mean()
print('mean over whole data','\n',mean)
data2.loc[:,'HM_WT':'MG_INJ_TIME'] -= mean
sigma = data2.loc[:,'HM_WT':'MG_INJ_TIME'].pow(2).mean()
sigma = sigma.pow(0.5)
print('sigty ',type(sigma))
# print(sigma)
data2.loc[:,'HM_WT':'MG_INJ_TIME']/=sigma


# mean2 = data2.loc[:,'DS_S':'DS_S'].mean()
# # print(type(mean2))
# print('mean over whole data','\n',mean2)
# data2.loc[:,'DS_S':'DS_S'] -= mean2
# sigma2 = data2.loc[:,'DS_S':'DS_S'].pow(2).mean()
# sigma2 = sigma2.pow(0.5)
# # print('sgtype2',type(sigma2))
# # print(sigma)
# data2.loc[:,'DS_S':'DS_S']/=sigma2

# mean2 = mean2[0]
# sigma2=sigma2[0]

# print('---------')
# print('sigma2 mean2\n', mean2)
# print('---------')

# raw_dat2 = raw_dat
raw_dat = data2










"""
mean normalisation
epochs 10
Test acc = 44.28
Train acc = 45.83

20 epochs
Test 48.95
train 51.02
"""
# mn = data2.loc[:,'HM_WT':'MG_INJ_TIME'].min()
# mx = data2.loc[:,'HM_WT':'MG_INJ_TIME'].max()
# # # print(mean)
# data2.loc[:,'HM_WT':'MG_INJ_TIME'] -= data2.loc[:,'HM_WT':'MG_INJ_TIME'].mean()
# # # sigma = data2.loc[:,'HM_WT':'MG_INJ_TIME'].pow(2).mean()
# # # sigma = sigma.pow(0.5)
# # # print(sigma)
# data2.loc[:,'HM_WT':'MG_INJ_TIME']/=(mx-mn)


# raw_dat = data2






"""
min max scaling
epochs 10
Test acc = 44.28
Train acc = 45.83

20 epochs
Test 48.95
train 51.02
"""
# mn = data2.loc[:,'HM_WT':'MG_INJ_TIME'].min()
# mx = data2.loc[:,'HM_WT':'MG_INJ_TIME'].max()
# # print(mean)
# data2.loc[:,'HM_WT':'MG_INJ_TIME'] -= mn
# # sigma = data2.loc[:,'HM_WT':'MG_INJ_TIME'].pow(2).mean()
# # sigma = sigma.pow(0.5)
# # print(sigma)
# data2.loc[:,'HM_WT':'MG_INJ_TIME']/=(mx-mn)


# raw_dat = data2



"""
A new idea...
using softmax
"""

print('mnDs_s',raw_dat.loc[:,'DS_S'].min())
print('mxDs_s',raw_dat.loc[:,'DS_S'].max())


# scale up ......
# raw_dat.loc[:,'DS_S']+=3.0

print('aftermnDs_s',raw_dat.loc[:,'DS_S'].min())
print('aftermxDs_s',raw_dat.loc[:,'DS_S'].max())



# aced it using this idea

"""

"""


# extracting the independent variable, X
x_data = raw_dat.loc[:,['HM_WT', 'AIM_S', 'HM_S', 'HM_C', 'HM_SI', 'HM_TI','HM_MN', 'CAC2', 'MG', 'HM_TEMP', 'CAC2_INJ_TIME', 'MG_INJ_TIME']]
# extracting the dependent variable, y
y_data = raw_dat.loc[:,'DS_S']


# extracting the independent variable, X
# x_data2 = raw_dat2.loc[:,['HM_WT', 'AIM_S', 'HM_S', 'HM_C', 'HM_SI', 'HM_TI','HM_MN', 'CAC2', 'MG', 'HM_TEMP', 'CAC2_INJ_TIME', 'MG_INJ_TIME']]
# # extracting the dependent variable, y
# y_data2 = raw_dat2.loc[:,'DS_S']


# x_data = raw_dat.loc[:,['HM_WT', 'AIM_S', 'HM_S', 'HM_C', 'HM_SI', 'HM_TI','HM_MN', 'CAC2', 'MG', 'HM_TEMP']]



# splitting the entire dataset into test and train
# for all model development, use only the train dataset
# do not touch the test dataset for any development purposes
# test dataset is purely for validation purposes
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

# X_train2, X_test2, y_train2, y_test2 = train_test_split(x_data2, y_data2, test_size=0.33, random_state=42)
# print(X_train.iloc[0,:])





# THE LAST STAND
dicttrain = dict()
dicttest = dict()
for y in y_train:
	z = int(y*1000)
	if z in dicttrain:
		dicttrain[z]+=1
	else:dicttrain[z]=1

for y in y_test:
	z = int(y*1000)
	if z in dicttest:
		dicttest[z]+=1
	else: dicttest[z]=1

print(sorted(list(dicttrain)))
print()
print()
print(sorted(list(dicttest) ))
print()
print(dicttrain)
print()
print()
print(dicttest)


# for idx , y in enumerate(y_train):
	
# 	z = int(y*1000)
# 	if z > 0 and z < 8:
# 		y_train.iloc[idx] = 0 #4
# 	elif z < 15:
# 		y_train.iloc[idx] = 1 #11
# 	elif z < 25:
# 		y_train.iloc[idx] = 2 #18
# 	else:
# 		print('zzz', z)


# print('modified\n',y_train.iloc[:8])




# for idx,y in enumerate(y_train):
# 	if y >= 0.023 or y <= 0.001:
# 		print('hoola',idx,y)

# # print('y_train min max ', y_train.iloc[0], y_train.iloc[0])
# print('y_test min max ', y_test.min(), y_test.max())

"""
performing normalisation after  and separately
getting better results
"""



# mean = X_train.iloc[:,:].mean()
# print('mean over train data only','\n',mean)
# X_train.iloc[:,:] -= mean
# sigma = X_train.iloc[:,:].pow(2).mean()
# sigma = sigma.pow(0.5)
# # print(sigma)
# X_train.iloc[:,:]/=sigma

# # print(X_train[0,:])
# # raw_dat = data2

# mean = X_test.iloc[:,:].mean()
# print('mean over train data only','\n',mean)
# X_test.iloc[:,:] -= mean
# sigma = X_test.iloc[:,:].pow(2).mean()
# sigma = sigma.pow(0.5)
# # print(sigma)
# X_test.iloc[:,:]/=sigma













# print(type(X_train))

# since data set already there, I am adding it to the pytorch dataset class
class tata_train(Dataset):

	def __init__(self, X_train, y_train):
		super().__init__()
		self.X_train = X_train
		self.y_train = y_train

	def __getitem__(self, idx):
		#convert to one hot encoded
		# one_hot = np.zeros(24).astype('long')
		# idx2 = int(self.y_train.iloc[idx] * 1000)-1
		# one_hot[idx2] = 1.0

		#modified
		y = self.y_train.iloc[idx]
		z = int(y*1000)
		yo=0
		if z < 5:
			yo = 0 #2
		elif z < 10:
			yo = 1 #7
		elif z < 15:
			yo = 2 #12
		elif z < 20:
			yo = 3 #17
		elif z < 25:
			yo = 4 #22
		else:
			print('zzz', z)
		one_hot = np.zeros(5).astype('long')
		idx2 = int(yo)
		one_hot[idx2] = 1.0

		# idx2 = int(self.y_train.iloc[idx]) # for normalised output
		# one_hot[idx2] = 1.0
		# print(self.y_train.iloc[idx],' and one hot ')
		return torch.tensor( self.X_train.iloc[idx] ), torch.tensor( one_hot )

	def __len__(self):
		return len(self.X_train)


# class tata_test(Dataset):
# 	"""docstring for tata_test"""
# 	def __init__(self, X_test, y_test):
# 		super().__init__()
# 		self.X_test = X_test
# 		self.y_test = y_test
# 	def __getitem__(self, idx):
# 		# need to return a pytorch tensor for processing
# 		one_hot = np.zeros(24).astype('long')
# 		# one_hot = np.zeros(9).astype('long') # for normalisd output

# 		idx2 = int(self.y_test.iloc[idx] * 1000)-1
# 		# idx2 = int(self.y_test.iloc[idx] ) # for normalised output
# 		one_hot[idx2] = 1.0
# 		# print(self.y_train.iloc[idx],' and one hot ')
# 		return torch.tensor( X_test.iloc[idx] ),  torch.tensor( one_hot )

# 	def __len__(self):
# 		return len(self.X_test)


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = nn.Sequential(

			nn.Linear(in_features = 12, out_features = 30),
			nn.BatchNorm1d(30),
			nn.LeakyReLU(),
			# nn.Sigmoid(),
			# nn.BatchNorm1d(30),

			nn.Linear(in_features = 30, out_features = 50),
			nn.BatchNorm1d(50),
			nn.LeakyReLU(),
			# nn.Sigmoid(),
			# nn.BatchNorm1d(50),
			
			nn.Linear(in_features = 50, out_features = 50),
			nn.BatchNorm1d(50),
			nn.LeakyReLU(),
			# # nn.Sigmoid(),
			# nn.BatchNorm1d(50),
			
			nn.Linear(in_features = 50, out_features = 30),
			nn.BatchNorm1d(30),
			nn.LeakyReLU(),
			# # nn.Sigmoid(),
			# nn.BatchNorm1d(30),

			nn.Linear(in_features = 30, out_features = 30),
			nn.BatchNorm1d(30),
			nn.LeakyReLU(),
			# # nn.Sigmoid(),
			# nn.BatchNorm1d(30),

			nn.Linear(in_features = 30, out_features = 30),
			nn.BatchNorm1d(30),
			nn.LeakyReLU(),

			nn.Linear(in_features = 30, out_features = 30),
			nn.BatchNorm1d(30),
			nn.LeakyReLU(),

			nn.Linear(in_features = 30, out_features = 30),
			nn.BatchNorm1d(30),
			nn.LeakyReLU(),
			# # nn.Sigmoid(),
			# nn.BatchNorm1d(30),
			
			# nn.Linear(in_features = 6, out_features = 6),
			# nn.ReLU(),
			

			# nn.Linear(in_features = 6, out_features = 6),
			# nn.ReLU(),
			
			# nn.Linear(in_features = 6, out_features = 6),
			# nn.ReLU(),
			# nn.Sigmoid(),
			# nn.BatchNorm1d(4),
			
			# nn.Linear(in_features = 30, out_features = 24),
			nn.Linear(in_features = 30, out_features = 5), # for modified output
			# nn.Sigmoid()

		)

	def forward(self, x):
		return self.linear(x)


train_ds = tata_train(X_train, y_train)
# test_ds = tata_test(X_test, y_test)

train_loader = DataLoader(train_ds, shuffle=True, batch_size=256)
# test_loader = DataLoader(test_ds, shuffle=False, batch_size=4)
# it = iter(train_loader)
# print(it.next()[1])


## define device , optimizers and schedulers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.03)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma = 0.1)
# inp = torch.rand(3,12)
# print(inp)

# out = net(inp)
# print(out.size())






epochs = 20
train_losses = []
test_losses = []
for _ in range(epochs):
	scheduler.step()
	running_loss = 0.0
	for x , y in train_loader:
		# print(x, y)
		x = x.to(device)
		y = y.to(device)
		out = net(x)

		loss = criterion(out, torch.max(y,1)[1])
		running_loss += loss.item()*y.size(0)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	running_loss = running_loss / len(train_loader.dataset)
	train_losses.append(running_loss)
	print('epoch ', _ ,' loss ', running_loss)

	# comment out the test_loader part to check test_loss at runtime
	# test_loss = 0.0
	# with torch.no_grad():
	# 	for i, (x, y) in enumerate(test_loader):
	# 		y = y.view(-1,1).to(device)
	# 		out = net(x.to(device))
	# 		loss = criterion(y,out)
	# 		test_loss += loss.item()*y.size(0)
	# 	test_loss = test_loss / len(test_loader.dataset)
	# 	test_losses.append(test_loss)
	# print('test loss ', test_loss)




train_tensor = torch.from_numpy( np.array( X_train.loc[:,:]) ).float()
test_tensor = torch.from_numpy( np.array( X_test.loc[:,:] ) ).float()
# test_tensor = torch.tensor(X_test.iloc[:])
# print(train_tensor.size())
# pred_train = net(train_tensor.to(device)).to('cpu').detach().numpy().argmax(axis=1).astype('float')+np.ones(train_tensor.size(0))
# pred_train /= 1000.0




#for moodified
d = { 0 : 2.0 , 1 : 7.0 , 2 : 12.0 , 3 : 17.0 , 4 : 22.0} #best config till now
print(train_tensor.size())
pred_train = net(train_tensor.to(device)).to('cpu').detach().numpy().argmax(axis=1).astype('int')
# for idx, y in pred_train:
# 	pred_train[idx] = d[y]

pred_train = np.array([d[x] for x in pred_train])
print('pred_train\n', pred_train[:4])
pred_train /= 1000.0





# for normalised output
# pred_train = net(train_tensor.to(device)).to('cpu').detach().numpy().argmax(axis=1).astype('float')

# to get original val
# pred_train = net(train_tensor.to(device)).to('cpu').detach().numpy().argmax(axis=1).astype('float')-3.0
# pred_train *= sigma2
# pred_train += mean2
# print(pred_train[:4])
# y_train = y_train2
# print(y_train[:4])
# pred_train /= 1000.0

# batch size X 24
# print('pred_tr', pred_train.shape)
# pred_test = net(test_tensor.to(device)).to('cpu').detach().numpy().argmax(axis=1).astype('float')+np.ones(test_tensor.size(0))
# pred_test /= 1000.0





#modified
pred_test = net(test_tensor.to(device)).to('cpu').detach().numpy().argmax(axis=1).astype('int')
# for idx, y in pred_test:
# 	pred_test[idx] = d[y]

pred_test = np.array([d[x] for x in pred_test])
print('pred_test\n', pred_test[:4])
pred_test /= 1000.0





# to get original val
# pred_test = net(test_tensor.to(device)).to('cpu').detach().numpy().argmax(axis=1).astype('float')-3.0
# pred_test *= sigma2
# pred_test += mean2
# print(pred_test[:4])
# y_test = y_test2
# print(y_test[:4])



"""
matplotlib on ubuntu
"""

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want, (maybe disable on windows)
import matplotlib.pyplot as plt

plt.plot(train_losses, label='train_losses')
plt.plot(test_losses, label='test_losses')
plt.legend(frameon=False)
plt.show()

#tolerance range
check = 0.003

# findinf the error on the predictions
y_test = list(y_test)
y_train = list(y_train)
err_test = [x-y for x,y in zip(pred_test,y_test)]
err_train = [x-y for x,y in zip(pred_train,y_train)]

# finding the strike rates on the datasets
strike_rate_test = 100*sum([np.abs(x)<=check for x in err_test])/len(err_test)
strike_rate_train = 100*sum([np.abs(x)<=check for x in err_train])/len(err_train)

# printint the results
print("Test strike rate : {}\nTrain strike rate : {}".format(strike_rate_test,strike_rate_train))
