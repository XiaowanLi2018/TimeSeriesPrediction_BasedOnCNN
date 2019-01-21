# TimeSeriesPrediction_BasedOnCNN
引言：
--
背景
-
<br>
一般来说，RNN作为时许预测是比较通用的，但是RNN有一些弊端，比如RNN耗时太长，由于网络一次只读取、解析输入的一个单位，深度神经网络必须等前一个单位处理完，才能进行下一个单位的处理。这意味着RNN不能像CNN那样进行大规模并行处理，且在实际操作层面，cnn的训练相对来说是比较容易训练的，且针对某些数据集，cnn的预测结果也可以达到和rnn一样好的效果，可以以因果卷积为基础来做时序预测

目标
-
* 实现因果卷机为基础的预测模型<br>
* 以因果卷积为基础进行改善，避免训练中出现的问题，提高预测准确率<br>
* 基于时序数据来对比各种网络结构的效果以及建模过程的注意点总结

术语及定义
-

1.因果卷积：<br>
时序预测要求对时刻t 的预测yt只能通过t时刻之前的输入x1到xt-1来判别（像隐马尔科夫链）,这在CNN里面就叫做因果卷积（causalconvolutions）
2.扩张卷积：<br>
扩展卷积是在普通卷机的基础上引入一个新的 hyper-parameter, dilate（扩张系数）, 这个 hyper-parameter 的涵义是每隔 dilate-1 个像素取一个” 像素”, 做卷积操作，扩张卷积可以做到每一层隐层都和输入序列大小一样，并且计算量降低，感受野足够大。
![](https://github.com/ZhouYuxuanYX/Wavenet-in-Keras-for-Kaggle-Competition-Web-Traffic-Time-Series-Forecasting/blob/master/figures/wavenet.gif)

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
1.base_wavenet模型仅使用最近的 k 个输入，即 x_t-k + 1，...，x_t 来预测 y_t，而不是依赖整个历史状态进行预测,这对应于强条件独立性假设。特别是，前馈模型假定目标仅取决于 k 个最近的输入
![](https://github.com/ZhouYuxuanYX/Wavenet-in-Keras-for-Kaggle-Competition-Web-Traffic-Time-Series-Forecasting/blob/master/figures/wavenet.gif)
<br>
2.为了提高准确率，还加入了残差卷积的跳层连接，以及1×1的卷积(TCN)



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
