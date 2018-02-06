#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
Data=pd.read_csv('D:/problem28-1.csv')
Data.index=Data.iloc[:,6]
Data.index=pd.to_datetime(Data.index,format='%Y-%m-%d')
Data=Data.iloc[:,0:4]
Data.head(2)

#提取收盘价
Close=Data.Close
Close.describe()

#求滞后1期的收盘价变量
lag1Close=Close.shift(1)

#求1日动量
momentum5=Close-lag1Close
momentum5.tail()

#绘制收盘价和1日动量曲线图
plt.rcParams['font.sans-serif']=['SimHei']
plt.subplot(211)
plt.plot(Close,'b*')
plt.xlabel('date')
plt.ylabel('Close')
plt.title('股价1日动量图')
plt.subplot(212)
plt.plot(momentum5,'r-*')
plt.xlabel('date')
plt.ylabel('Momentum5')

