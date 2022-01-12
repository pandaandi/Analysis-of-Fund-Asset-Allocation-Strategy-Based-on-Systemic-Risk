%%%%Q1
load R.mat
jz=corrcoef(R);
jz2=abs(jz);
imagesc(jz2);
colorbar
%老师！你的colorbar出来的图可能跟我们的图不是一个颜色，我们的是summer图样，可以从编辑-颜色图-工具-标准颜色图-summer路径设置
