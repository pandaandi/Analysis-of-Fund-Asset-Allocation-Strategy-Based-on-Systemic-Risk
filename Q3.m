stockdata=load('D:\matlabwork100fen\stockdata2.mat');
datedata=load('D:\matlabwork100fen\datedata.mat');
date_data=datedata.datedata(:,1);
%������ʷ��������
old_date=date_data./1000;
%����2020����������
new_date=date_data./1000;
%�����Ʊ����
stockprice_1=stockdata.stockdata.SZ(:,1);
stockprice_2=stockdata.stockdata.SZ(:,2);
stockprice_3=stockdata.stockdata.SZ(:,3);
stockprice_4=stockdata.stockdata.SZ(:,4);
stockprice_5=stockdata.stockdata.SZ(:,5);
stockprice_6=stockdata.stockdata.SZ(:,6);
stockprice_7=stockdata.stockdata.SZ(:,7);
stockprice_8=stockdata.stockdata.SZ(:,8);
stockprice_9=stockdata.stockdata.SZ(:,9);
stockprice_10=stockdata.stockdata.SZ(:,10);
stockprice_11=stockdata.stockdata.SZ(:,11);
stockprice_12=stockdata.stockdata.SZ(:,12);
stockprice_13=stockdata.stockdata.SZ(:,13);
stockprice_14=stockdata.stockdata.SZ(:,14);
stockprice_15=stockdata.stockdata.SZ(:,15);
stockprice_16=stockdata.stockdata.SZ(:,16);
stockprice_17=stockdata.stockdata.SZ(:,17);
%���ƹ�Ʊ�۸�����ͼ
%subplot(4,1,1);
plot(new_date,stockprice_1,new_date,stockprice_2,new_date,stockprice_3,new_date,stockprice_4,new_date,stockprice_5,new_date,stockprice_6,new_date,stockprice_7,new_date,stockprice_8,new_date,stockprice_9,new_date,stockprice_10,new_date,stockprice_11,new_date,stockprice_12,new_date,stockprice_13,new_date,stockprice_14,new_date,stockprice_15,new_date,stockprice_16,new_date,stockprice_17);
xlabel('date');
ylabel('stockprice');
%����ÿֻ��Ʊ����N
N=1000;
value=(stockprice_1+stockprice_2+stockprice_3+stockprice_4+stockprice_5+stockprice_6+stockprice_7+stockprice_8+stockprice_9+stockprice_10+stockprice_11+stockprice_12+stockprice_13+stockprice_14+stockprice_15+stockprice_16+stockprice_17) * N;
ret=price2ret(value);
subplot(4,1,2);
plot(new_date,value);
xlabel('time');
ylabel('value');
title('invest value');
subplot(2,1,2);
plot(new_date(2:end),ret,'*');
xlabel('time');
ylabel('returns');
title('Ͷ�������������')
figure;
histogram(ret,20);
ylabel('days');
xlabel('investment portfolio returns(per day)');
title('��ʷģ�ⷨͶ�������������ֱ��ͼ');
%��95%���Ŷ�ʱ����ֵ�����������
Var=prctile(ret,95) * value(end);
disp(['��ʷģ�ⷨͶ�����VaRΪ',num2str(Var)]);