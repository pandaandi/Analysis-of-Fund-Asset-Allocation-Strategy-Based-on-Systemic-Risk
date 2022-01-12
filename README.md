# Analysis-of-Fund-Asset-Allocation-Strategy-Based-on-Systemic-Risk
one of the assignments of the course Matlab Basics and Applications

这篇建模论文的选题来自于我们本学期参加的五一数学建模比赛的选题，在五一建模比赛中我们全程用的是Python，所以不存在任何套用行为。基于原论文框架下，我们使用MATLAB重新编译以及建模，仅在问题二的优化部分运用Python的Scipy模块实现。

另外本代码文件实现的过程，有几个注意事项（因学艺不精，细节问题需要手动操作）：
1、请把我们发过去的rar解压到D盘下面，保证路径为D:\matlabwork100fen，并按路径将其在MATLAB中打开；

2、代码部分需要您按顺序打开不同m文件并运行：Q0.m→Q1.m→Q2.m→Q3.m→Q4.m（代码放到一起只会显示出一张fig需要关闭一张之后才会出现下一张，或者图片显示不完整，比如三部分的画布只显示出两个部分，另一部分是空白）

3、Q0.m是用来下载股票数据的脚本，脚本里的token是Tushare注册后就有的使用码，可能会过期，我已经换上最新的token码了，如果还是不能运行，那么您可以查看我们发给您的文件夹（matlan100fen）中的文件夹（2019年SZ股票下载数据）中的17个xls文件，这17个xls文件是我们用Q0.m脚本下载出的17支股票在2019年的数据的副本。
%%并且向大家请教一个问题：在Q0脚本运行后，为什么下载到000021.SZ后就出错不能继续下载了？原以为是因为tushare积分限制，但是标红说是不能使用{}符号，那为什么前17支股票就可以进行下载呢？（下载到000021.sz但是只有17个代码是因为已经筛除部分无效股票）
%%当工作文件夹或其子文件夹中已经存在【2019年 000001.SZ 数据.xls】等文件时，Q0.m脚本会直接出错，这就是我们在子文件夹（2019年SZ股票下载数据）中所有xls文件的名称中都有个副本的原因，这样您才可以顺利运行。

4、数据在筛选和整合后被手动导入到MATLAB形成mat文件（已包含在文件夹内），在脚本运行过程中直接调用。

5、优化部分的爬虫代码也可以运行一下看看，脚本文件名为matlabyouhua.py，我用的是pycharm编译器。此爬虫脚本要下载很多package，您仔细看一下都安装后就可以运行啦。另外如果您要把py脚本调用文件的路径改掉，那么要特别注意路径中的所有\要改成\\，\会报错，\\就不会。

The topic selection of this modeling paper comes from the topic selection of the May 1 Mathematical Modeling Contest we participated in this semester. In the May 1 Modeling Contest, we used Python throughout the whole process, so there was no application behavior.Based on the framework of the original paper, we use MATLAB to recompile and model, and only use Python Scipy module to realize the optimization part of problem two.
In addition, the process of this code file implementation, there are several matters needing attention (due to the lack of fine learning, details need to be manually operated) :
1. Please decompress the RAR we sent to disk D, ensure the path is D:\matlabwork100fen, and open it in MATLAB according to the path;

2. The code part requires you to open different M files and run them in sequence: Q0.m→Q1. M →Q2.

3. Q0.m is the script used to download stock data. The token in the script is the usage code existing after Tushare registration, which may expire.Then you can check the 17 XLS files in the folder (2019 SZ stock download data) in the folder we sent you (Matlan100fen). These 17 XLS files are copies of the data of the 17 stocks we downloaded with q0.m script in 2019.
%% and ask you a question: after Q0 script run, why download 000021.sz after error can not continue to download?I thought it was because of the Tushare points limit, but the red flag said that the {} symbol can not be used, so why the first 17 stocks can be downloaded?(Downloaded to 000021.sz but only 17 codes because some invalid stocks have been screened out)
The q0.m script will directly fail when files like [2019 000001.sz data.xls] already exist in the working folder or its subfolders, which is why we have a copy of all the XLS files in the subfolder (2019 SZ Stock Download Data) in the name, so that you can run smoothly.

4. After screening and integration, the data is manually imported into MATLAB to form mat files (included in folders), which can be directly called in the process of script running.

5. The optimized crawler code can also be run. The script file is named matlabyouhua.py, and I used the PyCharm compiler.This crawler script will download a number of packages, and you can run them after you look at them carefully.In addition, if you want to change the path of the py script call file, be careful that all \ in the path are changed to \\, \ will report an error, \\ will not.
