stockdata=load('D:\matlabwork100fen\stockdata2.mat');
datedata=load('D:\matlabwork100fen\datedata.mat');
%绘制有效前沿曲线之前的基本操作与问题二一致
R=price2ret(stockdata.stockdata.SZ,[],'simple');
ExpReturn=nanmean(R);
ExpCov=nancov(R);
M=1000;N=17;
X1=zeros(M,N);
for i=1:M
    X1(i,:)=rand(1,N);
    X1(i,:)=X1(i,:)/sum(X1(i,:));
end
X2=zeros(M,N);
for i=1:M
    X2(i,:)=rand(1,N);
    X2(i,:)=X2(i,:)/sum(X2(i,:));
end
X3=zeros(M,N);
for i=1:M
    X3(i,:)=rand(1,N);
    X3(i,:)=X3(i,:)/sum(X3(i,:));
end
X4=zeros(M,N);
for i=1:M
    X4(i,:)=rand(1,N);
    X4(i,:)=X4(i,:)/sum(X4(i,:));
end
X5=zeros(M,N);
for i=1:M
    X5(i,:)=rand(1,N);
    X5(i,:)=X5(i,:)/sum(X5(i,:));
end
X6=zeros(M,N);
for i=1:M
    X6(i,:)=rand(1,N);
    X6(i,:)=X6(i,:)/sum(X6(i,:));
end
X7=zeros(M,N);
for i=1:M
    X7(i,:)=rand(1,N);
    X7(i,:)=X7(i,:)/sum(X7(i,:));
end
X8=zeros(M,N);
for i=1:M
    X8(i,:)=rand(1,N);
    X8(i,:)=X8(i,:)/sum(X8(i,:));
end
X9=zeros(M,N);
for i=1:M
    X9(i,:)=rand(1,N);
    X9(i,:)=X9(i,:)/sum(X9(i,:));
end
X10=zeros(M,N);
for i=1:M
    X10(i,:)=rand(1,N);
    X10(i,:)=X10(i,:)/sum(X10(i,:));
end
Weights=[X1;X2;X3;X4;X5;X6;X7;X8;X9;X10];
for i=1:10000
[PortRisk(i),PortReturn(i)]=portstats(ExpReturn,ExpCov,Weights(i,:));
end
p = Portfolio('assetmean', ExpReturn, 'assetcovar', ExpCov, 'lowerbudget', 1, 'upperbudget', 1, 'lowerbound', 0);
plot(PortRisk, PortReturn,'b.')
plotFrontier(p);
%计算投资组合最优风险区域
weights = estimateFrontierByRisk(p,PortRisk);
[risk1,ret1] = estimatePortMoments(p,weights);
%对比计算投资组合最优回报区域
weights = estimateFrontierByReturn(p,PortReturn);
[risk2,ret2] = estimatePortMoments(p,weights);
hold on
%绘制图像
plot(risk1,ret1,'-g',risk2,ret2,'*r');
