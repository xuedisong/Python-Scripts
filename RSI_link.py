# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

#获取黄金交易数据
Gold = pd.read_csv("gold.csv")
Gold.index = Gold.iloc[:,0]
Gold.index = pd.to_datetime(Gold.index,format='%Y-%m-%d')
Gold = Gold.iloc[:,1:]
Gold.head()
#构造数据框
price = Gold.close

'''
pChange = price - price.shift(1)
pChange = pChange.dropna()#去除NaN值
pChange[0:6]
indexP = pChange.index
up = pd.Series(0,index = indexP)#价格上涨
up[pChange > 0] = pChange[pChange > 0]
down = pd.Series(0,index = indexP)#价格下跌
down[pChange < 0] = -pChange[pChange < 0]
rsiframe = pd.concat([price,pChange,up,down],axis = 1)#按列合并
rsiframe.columns = ['price','pChange','up','down']
rsiframe = rsiframe.dropna()
#用简单平均数，计算RSI
sup = []
sdown = []
for i in range(6,len(up)+1):
    sup.append(np.mean(up.values[(i-6):i],dtype = np.float32))
    sdown.append(np.mean(down.values[(i-6):i],dtype = np.float32))
#计算6日RSI的值
rsi6 = [100*sup[i]/(sup[i]+sdown[i]) for i in range(0,len(sup))]
rsi6 = pd.Series(rsi6,index=indexP[5:])
sup = pd.Series(sup,index=indexP[5:])
sdown = pd.Series(sdown,index=indexP[5:])
'''

def rsi(price,period=6):    
    pChange = price - price.shift(1)
    pChange = pChange.dropna()#去除NaN值
    pChange[0:6]
    indexP = pChange.index
    up = pd.Series(0,index = indexP)#价格上涨
    up[pChange > 0] = pChange[pChange > 0]
    down = pd.Series(0,index = indexP)#价格下跌
    down[pChange < 0] = -pChange[pChange < 0]
    rsiframe = pd.concat([price,pChange,up,down],axis = 1)#按列合并
    rsiframe.columns = ['price','pChange','up','down']
    rsiframe = rsiframe.dropna()
    #用简单平均数，计算RSI
    sup = []
    sdown = []
    for i in range(period,len(up)+1):
        sup.append(np.mean(up.values[(i-period):i],dtype = np.float32))
        sdown.append(np.mean(down.values[(i-period):i],dtype = np.float32))
    #计算period日RSI的值
    rsi = [100*sup[i]/(sup[i]+sdown[i]) for i in range(0,len(sup))]
    rsi = pd.Series(rsi,index=indexP[(period-1):])
    sup = pd.Series(sup,index=indexP[(period-1):])
    sdown = pd.Series(sdown,index=indexP[(period-1):])
    return(rsi)

rsi6 = rsi(price,6)
rsi12 = rsi(price,12)