load SZ.mat
R1=tick2ret(SZ,[],'Simple');  
%求这十七只股票的收益率矩阵
expr=mean(R1);              
%求收益率均值，发现存在NaN
EXPR=nanmean(R1);           
%优化程序，得到期望收益率矩阵
expcov=cov(R1);
%求收益率的协方差，同样发现存在NaN             
ExpCov=nancov(R1);
%优化程序
M=1000;N=17;
X1=zeros(M,N);
for i=1:M
    X1(i,:)=rand(1,N)
    X1(i,:)=X1(i,:)/sum(X1(i,:));
end
M=1000;N=17;
X5=zeros(M,N);
for i=1:M
    X5(i,:)=rand(1,N)
    X5(i,:)=X5(i,:)/sum(X5(i,:));
end
X8=zeros(M,N);
for i=1:M
    X8(i,:)=rand(1,N)
    X8(i,:)=X8(i,:)/sum(X8(i,:));
end
X2=zeros(M,N);
for i=1:M
    X2(i,:)=rand(1,N)
    X2(i,:)=X2(i,:)/sum(X2(i,:));
end
load X3.mat
load X4.mat
load X6.mat
load X7.mat
load X9.mat
load X10.mat
Weights=[X1;X2;X3;X4;X5;X6;X7;X8;X9;X10];
%由于生成10000*17的矩阵需要耗费大量时间，所以这里采用随机生成10组1000*17的矩阵，然后将他们组合成一个新的10000*17的矩阵，可以较快地得到10000组权重。
for i=1:10000
[PortRisk(i),PortReturn(i)]=portstats(EXPR,ExpCov,Weights(i,:));
end
%分别计算这17支股票的风险收益
plot(PortRisk,PortReturn,'r.')
%绘制出风险收益散点图
p = Portfolio('assetmean', EXPR, 'assetcovar', ExpCov, 'lowerbudget', 1, 'upperbudget', 1, 'lowerbound', 0);
plotFrontier(p)
%画出有效前沿曲线
weights = estimateMaxSharpeRatio(p);
[risk, ret] = estimatePortMoments(p, weights);
hold on
plot(risk,ret,'*r');