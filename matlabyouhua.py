##老师，我用的是pycharm，然后这个代码有很多包要下载，注意一下
##我自己实验过两遍，是可以运行的哦
# coding=utf-8
import pandas as pd
from pandas import read_excel
import matplotlib.pyplot as plt
import random
import numpy as np
import xlrd
from datetime import date, datetime
import openpyxl

#图像中正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

with pd.ExcelFile('D:\\matlabwork100fen\\逆透视数据.xlsx') as xls:  # 将包含多个工作表的excel读取为一个文件夹
    stockprice = pd.read_excel(xls, 'Sheet1')  # 读取单个工作表

ts_code_list = ['ts_code01', 'ts_code02', 'ts_code03', 'ts_code04', 'ts_code05', 'ts_code06', 'ts_code07', 'ts_code08',
               'ts_code09', 'ts_code10', 'ts_code11', 'ts_code12', 'ts_code13', 'ts_code14', 'ts_code15', 'ts_code16',
               'ts_code17']

# 数据透视成二维
stock2 = stockprice.pivot(index='trade_date', columns='ts_code')['close']
print(stock2)

# 利用shift计算每日收益
#想看优化前就把下面这行代码注释掉，然后把此行代码的注释部分删除#stockreturn = stock2 / stock2.shift(1) - 1
stockreturn = np.log(stock2/stock2.shift(1))#优化后

#stockreturn.iloc[0:243, 0:57]

stockreturn.fillna(0, inplace=True)

# stockreturn.iloc[0:243,0:17]
print(stockreturn)
#stockreturn.to_excel('每日收益数据.xlsx')

# 年化收益
# stockreturn.mean()* 252

# 将收益率数据拷贝到新的变量 stock_return 中，这是为了后续调用的方便
#stock_return = stockreturn.copy()

# 计算协方差矩阵
cov_mat = stockreturn.cov()
# 年化协方差矩阵
cov_mat_annual = cov_mat * 252
# 输出协方差矩阵
print(cov_mat_annual)

# 蒙特卡洛模拟Markowitz模型
# 设置模拟的次数
number = 10000
# 设置空的numpy数组，用于存储每次模拟得到的权重、收益率和标准差
random_p = np.empty((number, 19))  # 19=17+2
# 设置随机数种子，这里是为了结果可重复
np.random.seed(19)

# 循环模拟10000次随机的投资组合
for i in range(number):
    # 生成17个随机数，并归一化，得到一组随机的权重数据
    random17 = np.random.random(17)
    random_weight = random17 / np.sum(random17)

    # 计算年平均收益率
    mean_return = stockreturn.mul(random_weight, axis=1).sum(axis=1).mean()
    #annual_return = (1 + mean_return) ** 252 - 1
    annual_return = np.sum(stockreturn.mean()*random_weight)*252

    # 计算年化标准差，也成为波动率
    random_volatility = np.sqrt(np.dot(random_weight.T, np.dot(cov_mat_annual, random_weight)))

    # 将上面生成的权重，和计算得到的收益率、标准差存入数组random_p中
    random_p[i][:17] = random_weight
    random_p[i][17] = annual_return
    random_p[i][18] = random_volatility

# 将Numpy数组转化为DataF数据框
RandomPortfolios = pd.DataFrame(random_p)
# RandomPortfolios.iloc[0:57,0:59]
#RandomPortfolios.fillna(0, inplace=True)
# 设置数据框RandomPortfolios每一列的名称
RandomPortfolios.columns = [ts_code + '_weight' for ts_code in ts_code_list] + ['Returns', 'Volatility']
#print(RandomPortfolios)

# 绘制散点图
##plt.show()
##RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)


# 找到标准差最小数据的索引值
min_index = RandomPortfolios.Volatility.idxmin()

# 在收益-风险散点图中突出风险最小的点,标准差为0

RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
plt.show()

x = RandomPortfolios.loc[min_index, 'Volatility']
y = RandomPortfolios.loc[min_index, 'Returns']

plt.scatter(x, y, color='red')
# 将该点坐标显示在图中并保留四位小数
plt.text(np.round(x, 4), np.round(y, 4), (np.round(x, 4), np.round(y, 4)), ha='left', va='bottom', fontsize=10)
plt.show()

# 提取最小波动组合对应的权重, 并转换成Numpy数组
##GMV_weights = np.array(RandomPortfolios.iloc[min_index, 0:57])
GMV_weights = RandomPortfolios.iloc[min_index, 0:17]
# 计算GMV投资组合收益
#stockreturn['Portfolio_GMV'] = stockreturn.mul(GMV_weights, axis=1).sum(axis=1)
#输出风险最小投资组合的权重
print(GMV_weights)


# 设置无风险回报率为0
risk_free = 0
# 计算每项资产的夏普比率
RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free) / RandomPortfolios.Volatility
# 绘制收益-标准差的散点图，并用颜色描绘夏普比率
plt.scatter(RandomPortfolios.Volatility, RandomPortfolios.Returns, c=RandomPortfolios.Sharpe)
plt.colorbar(label='Sharpe Ratio')
plt.show()

# 找到夏普比率最大数据对应的索引值
max_index = RandomPortfolios.Sharpe.idxmax()
# 在收益-风险散点图中突出夏普比率最大的点

x = RandomPortfolios.loc[max_index,'Volatility']
y = RandomPortfolios.loc[max_index,'Returns']
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
plt.scatter(x, y, color='red')
#将该点坐标显示在图中并保留四位小数
plt.text(np.round(x,4),np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10)
plt.show()


# 提取最大夏普比率组合对应的权重，并转化为numpy数组
##MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:57])
MSR_weights = RandomPortfolios.iloc[max_index, 0:17]
# 计算MSR组合的收益
##stockreturn['Portfolio_MSR'] = stockreturn.mul(MSR_weights, axis=1).sum(axis=1)
#输出夏普比率最大的投资组合的权重
print(MSR_weights)

umax_index=RandomPortfolios.idxmax()
x=RandomPortfolios.loc[umax_index, 'Volatility']
y=RandomPortfolios.loc[umax_index, 'Returns']
RandomPortfolios.plot('Volatility','Returns', kind='scatter',alpha=0.3)
plt.scatter(x,y,color='red')
plt.text(np.round(x,4),np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10)
plt.show()

Umax_weights = RandomPortfolios.iloc[umax_index, 0:17]
# 计算Umax组合的收益
##stockreturn['Portfolio_MSR'] = stockreturn.mul(MSR_weights, axis=1).sum(axis=1)
#输出效用最大的投资组合的权重
print(Umax_weights)

####

def statistics(random_weight):
    #根据权重，计算资产组合收益率/波动率/夏普率/效用函数。
    #输入参数
    #==========
    #random_weight : array-like 权重数组
    #权重为股票组合中不同股票的权重
    #返回值
    #=======
    #pret : float
    #      投资组合收益率
    #pvol : float
    #      投资组合波动率
    #pret / pvol : float
    #    夏普率，为组合收益率除以波动率，此处不涉及无风险收益率资产
    #

    #random_weight = np.array(random_weight)
    #pret = np.sum(rets.mean() * random_weight) * 252
#pvol = np.sqrt(np.dot(random_weight.T, np.dot(rets.cov() * 252, random_weight)))
    randomweights = np.array(random_weight)

    # 计算年平均收益率
    mean_return = stockreturn.mul(randomweights, axis=1).sum(axis=1).mean()
    #annual_return = (1 + mean_return) ** 252 - 1
    annual_return = np.sum(stockreturn.mean()*randomweights)*252

    # 计算年化标准差，也叫做波动率
    random_volatility = np.sqrt(np.dot(randomweights.T, np.dot(cov_mat_annual, randomweights)))

    return np.array([annual_return, random_volatility, annual_return / random_volatility, annual_return-(1/2)*4*random_volatility**2])

import scipy.optimize as sco

def min_func_sharpe(randomweights):
    return -statistics(randomweights)[2]

noa=17
bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',  bounds=bnds, constraints=cons)
#最大夏普率的投资组合的权重分配,精确到小数点后三位
print(opts['x'].round(3))
# 预期收益率约为, 预期被动率约为, 最优夏普指数，效用
print(statistics(opts['x']).round(3))

def min_func_variance(randomweights):
    return statistics(randomweights)[1]**2

optv = sco.minimize(min_func_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
#最小方差的投资组合的权重分配,精确到小数点后三位
print(optv['x'].round(3))
# 预期收益率约为, 预期被动率约为, 最优夏普指数,效用
print(statistics(optv['x']).round(3))


#效用最大
def min_func_utility(randomweights):
    return -statistics(randomweights)[2]

optu = sco.minimize(min_func_utility, noa * [1./ noa, ], method='SLSQP', bounds=bnds, constraints=cons)
print(optu['x'].round(3))
print(statistics(optu['x']).round(3))

##有效界
def min_func_port(randomweights):
    return statistics(randomweights)[1]


# 在不同目标收益率水平（ trets ）中循环时。 最小化的一个条件会变化。
# 这就是每次循环中更新条件字典对象的原因：

trets = np.linspace(0.0, 0.8, 100)
tvols = []
bnds = tuple((0, 1) for x in random_weight)
for tret in trets:
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1./ noa,], method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(8,4))
# random portfolio composition
plt.scatter(random_volatility,annual_return,c=annual_return/random_volatility,marker='o',alpha=0.3)
# efficient frontier
plt.scatter(tvols,trets,c=trets/tvols,marker='x')
# portfolio with highest Sharpe ratio
plt.plot(statistics(opts['x'])[1],statistics(opts['x'])[0],'r*',markersize=15.0)
# minimum variance portfolio
plt.plot(statistics(optv['x'])[1],statistics(optv['x'])[0],'y*',markersize=15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

import scipy.interpolate as sci

ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]

tck = sci.splrep(evols, erets)

# 通过这条数值化路径，最终可以为有效边界定义一个连续可微函数
# 和对应的一阶导数函数df(x):

def f(x):
    """
    Efficient frontier function (splines approximation)
    :param x:
    :return:
    """
    return sci.splev(x, tck, der=0)


def df(x):
    """
    First derivative of efficient frontier function.
    :param x:
    :return:
    """
    return sci.splev(x, tck, der=1)

