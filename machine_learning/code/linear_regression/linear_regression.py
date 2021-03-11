import csv
import pandas as pd
import numpy as np
import math

data = pd.read_csv("./train.txt")
# print(len(data))
# 4320 x 27

# 预处理
data = data.iloc[:,3:]
data[data=="NR"] = 0
raw_data = data.to_numpy()
# print(raw_data.shape)

# 提取特征
month_data = {}
for month in range(12):
    # 一个月
    sample = np.empty([18,480]) # 24小时x20天
    for day in range(20):
        sample[:,day*24:(day+1)*24] = raw_data[18*(month*20+day):18*(20*month+day+1),:]
    month_data[month] = sample

x = np.empty([12*471,18*9],dtype=float)
y = np.empty([12*471,1],dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month*471+day*24+hour,:] = month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,-1) # 数据变成了一行
            y[month*471+day*24+hour,0] = month_data[month][9,day*24+hour+9]

# 标准化
mean_x = np.mean(x,axis=0) # 对每一列
std_x = np.std(x,axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j]-mean_x[j]) / std_x[j]

# 将训练集划分为训练集和验证集
# math.floor:用于获取下限值
x_train_set = x[:math.floor(len(x)*0.8),:]
y_train_set = y[:math.floor(len(x)*0.8),:]

x_validation_set = x[math.floor(len(x)*0.8):,:]
y_validation_set = y[math.floor(len(x)*0.8):,:]

dim = 18*9 + 1 # b也占一个维度
w = np.zeros([dim,1]) # 随机初始化为0
x = np.concatenate((np.ones([12*471,1]),x),axis=1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim,1])
eps = 0.00000001
beta = 0.999
los = []
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x,w)-y,2))/471/12) # rmse
    los.append(loss)
    if t % 100 == 0:
        print(str(t)+":"+str(loss))
    gradient = 2*np.dot(x.transpose(),np.dot(x,w) - y)
    adagrad += gradient**2
    w = w - learning_rate*gradient / np.sqrt(adagrad+eps)
print(str(t)+":"+str(loss))
np.save("weight.npy",w)

# 导入测试数据
# 读入测试数据test.csv
testdata = pd.read_csv('./test.txt', header = None)
# 丢弃前两列，需要的是从第3列开始的数据
test_data = testdata.iloc[:, 2:]
# 把降雨为NR字符变成数字0
test_data[test_data == 'NR'] = 0
# 将dataframe变成numpy数组
test_data = test_data.to_numpy()
# 将test数据也变成 240 个维度为 18 * 9 + 1 的数据。
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

ans_y = np.dot(test_x, w)

with open("submit.csv",mode='w',newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id','value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_'+str(i),ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)