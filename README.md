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
时序预测要求对时刻t 的预测yt只能通过t时刻之前的输入x1到xt-1来判别（像隐马尔科夫链）,这在CNN里面就叫做因果卷积（causalconvolutions）<br>
2.扩张卷积：<br>
扩展卷积是在普通卷机的基础上引入一个新的 hyper-parameter, dilate（扩张系数）, 这个 hyper-parameter 的涵义是每隔 dilate-1 个像素取一个” 像素”, 做卷积操作，扩张卷积可以做到每一层隐层都和输入序列大小一样，并且计算量降低，感受野足够大<br>
![](https://github.com/XiaowanLi2018/TimeSeriesPrediction_BasedOnCNN/blob/master/data/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-01-21%20%E4%B8%8A%E5%8D%8810.10.31.png)
<br>
3.卷积运算<br>
![](https://github.com/XiaowanLi2018/TimeSeriesPrediction_BasedOnCNN/blob/master/data/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-01-21%20%E4%B8%8B%E5%8D%884.01.39.png)
![](https://github.com/XiaowanLi2018/TimeSeriesPrediction_BasedOnCNN/blob/master/data/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-01-21%20%E4%B8%8B%E5%8D%884.01.26.png)<br>
其中 d 为扩张系数、k 为卷积核大小，s − d · i 计算了采用上层哪一个单元。扩张系数控制了每两个卷积核间会插入多少零值，当 d=1 时，空洞卷积就会退化为一般的卷积运算。使用较大的扩张系数允许输出端的神经元表征更大范围的输入序列，因此能有效扩张感受野

算法设计：
-
<br>
1.整体来说经过多次尝试，对于wavenet这个网络来说一定程度的增大感受野会对提升准确率有一定帮助，但是wavenet整体网络区域训练过程中过拟合的情况，这里有几个可以显著提高准确率的设计：<br>
（1）保持teaching force不变，seq2seq中加入前一个时间切片的数据和原始的encode数据拼接起来作为输入会极大的提高准确率<br>
（2）在dilated conv加入dropout layer可有效地避免过拟合<br>
2.如果模型没能很好的学到趋势，可在dilated conv上堆叠，进而增大感受野<br>
3.为提高预测准确率，实现因果dilated conv+res block，是每隔一层跳层连接，每一个跳层连接块都是dilated+dropout+1*1conv的图示中的结构，降低模型复杂度<br>

数据输入阶段
-

- 训练阶段

| input                                     | output                      | description           |
| :---------------------------------------- | :-------------------------- | :-------------------- |
| (batch_size,encode_length,feature_length) |(batch_size,pred_length,1)   |encode_length:时序数据编码长度，feature_length:和时间点对应的特征长度，pred_length:要预测的时间长度，batch_size:每个batch送入训练的时间序列的条目数<br>,ps:此时input为编码长度和前一时间切片拼接后的array <br>

- 预测阶段

| input                                     | output                      |
| ----------------------------------------- | :-------------------------- |
| (batch_size,encode_length,feature_length) | (batch_size,pred_length,1)  |


算法网络：
--
1.base_wavenet模型仅使用最近的 k 个输入，即 x_t-k + 1，...，x_t 来预测 y_t，而不是依赖整个历史状态进行预测,这对应于强条件独立性假设。特别是，前馈模型假定目标仅取决于 k 个最近的输入<br>
![](https://github.com/ZhouYuxuanYX/Wavenet-in-Keras-for-Kaggle-Competition-Web-Traffic-Time-Series-Forecasting/blob/master/figures/wavenet.gif)
<br>
2.为了提高准确率，还加入了残差卷积的跳层连接，以及1×1的卷积(TCN)
![](https://github.com/XiaowanLi2018/TimeSeriesPrediction_BasedOnCNN/blob/master/data/Screenshot-from-2018-06-09-162900.png)
<br>
3.为了扩大感受野，帮助模型更好的学习特征，可以堆叠res_block+孔洞卷积（对于现有的dataset，暂时不需要堆叠来提升学习效果，网络过于庞大）<br>
![](https://github.com/XiaowanLi2018/TimeSeriesPrediction_BasedOnCNN/blob/master/data/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-01-21%20%E4%B8%8B%E5%8D%882.27.02.png)<br>

模型效果：
--
随机选择验证集中的2000条时间序列测试结果，其mae基本在0.05-0.11之间：<br>
![](https://github.com/XiaowanLi2018/TimeSeriesPrediction_BasedOnCNN/blob/master/data/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-01-21%20%E4%B8%8B%E5%8D%882.31.52.png)<br>
随机统计了mape其中1000条时间序列的单点的mape，其中一次的具体结果样例:<br>
![](https://github.com/XiaowanLi2018/TimeSeriesPrediction_BasedOnCNN/blob/master/data/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-01-21%20%E4%B8%8B%E5%8D%882.33.19.png)

使用方法以及说明：
--
**模型参数**

| hyper_parameter                  | description                 |
| -------------------------------- | :-------------------------- |
| n_filters                        | 卷积核的个数                  |
| filter_width                     | 卷积核的大小                  |
| dilation_rates                   | 扩张系数范围                  |
| padding                          | 卷积填充类型                  |
| dropout_rate                     | dropout系数                  |
| activation                       | 激活类型                     |
| use_skip_connections             | 是否用残差跳层                |
| return_sequence                  | 是否需要预测序列              |
| PRED_LENGTH                      | 预测序列的长度                |

**具体脚本功用**

1. 数据预处理<br>
trainer.py 数据读入以及数据集划分<br>
util.py 基本数据处理<br>
2. 训练<br>
train.py<br>
运行：python3 train.py<br>
3. 预测<br>
predict.py<br>
运行： python3 predict.py<br>
4. 模型<br>
my_models.py<br>

参考文献：
--
1.[CNN时序预测基础论文](https://arxiv.org/pdf/1803.01271.pdf）<br>
2.[Wavenet](https://github.com/ZhouYuxuanYX/Wavenet-in-Keras-for-Kaggle-Competition-Web-Traffic-Time-Series-Forecasting)<br>
3.[TCN](https://github.com/philipperemy/keras-tcn)
