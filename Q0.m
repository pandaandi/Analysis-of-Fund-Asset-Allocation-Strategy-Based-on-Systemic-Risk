close all
clear
clc
%%%1 在matlab中利用tushare读取深圳A股股票历史数据
f = '股票名单.xlsx';%将股票名单.xlsx放入matlabwork中，或者老师直接运行一遍然后放到红字路径
[~,stocklist] = xlsread(f);
addpath(genpath(pwd));
% token注册tushare后就有，可能会更新，但是所有下载出的文件都会放在附件里供老师参考
% 另外tushare里积分不足会限制下载，因此会在下载中途出红停止下载，望老师理解
token = '10e7246f79b33f1f8463932bd79c3a27fe9f37a8499f49d6a4144b90';
api = pro_api(token);
start_time = '20190101';
end_time = '20191231';
nstock = length(stocklist);
for i = 1:nstock
    df = api.query('daily', 'ts_code',stocklist{i}, 'start_date',start_time, 'end_date',end_time);
    data = flipud(df);
    writetable(data,['2019年 ',stocklist{i},' 数据.xls']);
end
%所有读取后的xls，手工处理汇集成了【2019年 SZ部分股票.xls】