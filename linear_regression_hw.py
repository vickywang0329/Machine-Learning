##linear regression
#由前9個小時的18個features(包含PM2.5)預測的10個小時的PM2.5
#load 'train.csv'

import sys
import pandas as pd
import numpy as np
data = pd.read_csv('D:\\train.csv', encoding='big5')
data= data.iloc[:,3:]
data[data=='NR']=0
raw_data= data.to_numpy()

month_data = {}
for month in range(12):
	sample = np.empty([18, 480])
	for day in range(20):
		sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month+ day) : 18 * (20 * month + day + 1), :]
		month_data[month] = sample

x = np.empty([12*471, 18*9], dtype=float)
y= np.empty([12*471,1], dtype=float)
for month in range(12):
	for day in range(20):
		for hour in range(24):
			if day == 19 and hour >14:
				continue
			x[month*471 + day*24 + hour ,:]= month_data[month][:, day*24+ hour: day*24 + hour + 9].reshape(1,-1)
			y[month*471 + day*24 + hour, 0]= month_data[month][9, day*24 + hour+9]
print(x)
print(y)

mean_x= np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
	for j in range(len(x[0])):
		if std_x[j]!=0:
			x[i][j]= (x[i][j]- mean_x[j])/ std_x[j]
print(x)

import math
x_trian_set = x[:math.floor(len(x)*0.8), :]
y_trian_set = y[: math.floor(len(y)*0.8), :]
x_validation = x[math.floor(len(x)*0.8):, :]
y_validation= y[math.floor(len(y)*0.8):,:]
print(x_trian_set)
print(y_trian_set)
print(x_validation)
print(y_validation)
print(len(x_trian_set))
print(len(y_trian_set))
print(len(x_validation))
print(len(y_validation))

#training
dim = 18*9+1
w= np.zeros([dim, 1])
x= np.concatenate((np.ones([12*471, 1]), x), axis=1).astype(float)
learning_rate= 100
iter_time =1000
adagrad= np.zeros([dim, 1])
eps= 0.0000000001
for t in range(iter_time):
	loss = np.sqrt(np.sum(np.power(np.dot(x,w)-y,2))/471/12)
	if(t%100==0):
		print(str(t)+ ":"+ str(loss))
	gradient = 2*np.dot(x.transpose(), np.dot(x,w)-y)
	adagrad+= gradient**2
	w= w-learning_rate* gradient/ np.sqrt(adagrad+eps)
np.save('weight.npy', w)
print(w)

#testing#載入test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理
#使test data 形成 240個維度為18*9+1

testdata = pd.read_csv('D:\\test.csv', header=None, encoding='big5')
test_data= testdata.iloc[:, 2:]
test_data[test_data=='NR']=0
test_data =test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype=float)
for i in range(240):
	test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
	for j in range(len(test_x[0])):
		if std_x[j]!= 0:
			test_x[i][j]= (test_x[i][j]-mean_x[j])/std_x[j]
test_x = np.concatenate((np.ones([240,1]), test_x), axis=1).astype(float)
print(test_x)

#prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
print(ans_y)

import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
