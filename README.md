# TimeSeriesPrediction_BasedOnCNN
引言：
--
背景
-
<br>
一般来说，RNN作为时许预测是比较通用的，但是RNN有一些弊端，比如RNN耗时太长，由于网络一次只读取、解析输入的一个单位，深度神经网络必须等前一个单位处理完，才能进行下一个单位的处理。这意味着RNN不能像CNN那样进行大规模并行处理，且在实际操作层面，cnn的训练相对来说是比较容易训练的，且针对某些数据集，cnn的预测结果也可以达到和rnn一样好的效果，可以以因果卷积为基础来做时序预测
目标
-
<br>
* 实现因果卷机为基础的预测模型<br>
* 以因果卷积为基础进行改善，避免训练中出现的问题，提高预测准确率<br>
* 基于时序数据来对比各种网络结构的效果以及建模过程的注意点总结
<br>
术语及定义
-
<br>
1.因果卷积：<br>
2.
算法设计：
--
<br>


数据输入阶段
-
<br>
训练阶段<br>
预测阶段
<br>
算法网络：
--
![](https://github.com/ZhouYuxuanYX/Wavenet-in-Keras-for-Kaggle-Competition-Web-Traffic-Time-Series-Forecasting/blob/master/figures/wavenet.gif)

模型效果：
--
<br>
使用方法以及说明：
--


针对时序数据的预测，现在目前多数会采用以rnn为基础的模型来进行，
--
     Wavenet
Optimizations:
--
     1. Change Net Structure (+LSTM/+Dropout) 
     2. Add Residual Block and try to build structure like TCN
