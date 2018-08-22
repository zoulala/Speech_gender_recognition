# Speech_gender_recognition
使用机器学习的方法判断一段音频信号是男性还是女性。

# 项目参考：
- https://github.com/nd009/capstone/tree/master/Gender_Recognition_by_Voice
- https://github.com/primaryobjects/voice-gender
- http://blog.topspeedsnail.com/archives/10729

# Data
voice.csv文件，数据来自[github](https://github.com/primaryobjects/voice-gender)

格式如下：
```
   meanfreq        sd    median       Q25       Q75       IQR       skew  \
0  0.059781  0.064241  0.032027  0.015071  0.090193  0.075122  12.863462
1  0.066009  0.067310  0.040229  0.019414  0.092666  0.073252  22.423285
2  0.077316  0.083829  0.036718  0.008701  0.131908  0.123207  30.757155
3  0.151228  0.072111  0.158011  0.096582  0.207955  0.111374   1.232831
4  0.135120  0.079146  0.124656  0.078720  0.206045  0.127325   1.101174

          kurt    sp.ent       sfm  ...    centroid   meanfun    minfun  \
0   274.402906  0.893369  0.491918  ...    0.059781  0.084279  0.015702
1   634.613855  0.892193  0.513724  ...    0.066009  0.107937  0.015826
2  1024.927705  0.846389  0.478905  ...    0.077316  0.098706  0.015656
3     4.177296  0.963322  0.727232  ...    0.151228  0.088965  0.017798
4     4.333713  0.971955  0.783568  ...    0.135120  0.106398  0.016931

     maxfun   meandom    mindom    maxdom   dfrange   modindx  label
0  0.275862  0.007812  0.007812  0.007812  0.000000  0.000000   male
1  0.250000  0.009014  0.007812  0.054688  0.046875  0.052632   male
2  0.271186  0.007990  0.007812  0.015625  0.007812  0.046512   male
3  0.250000  0.201497  0.007812  0.562500  0.554688  0.247119   male
4  0.266667  0.712812  0.007812  5.484375  5.476562  0.208274   male
```

# results
传统机器学习结果:
- SVM accuracy score: 0.967
- DT accuracy score: 0.965
- LR accuracy score: 0.914

xgboost:
- XGB accuracy score: 0.986

深度神经网络结果：
- MLP accuracy score: 0.979

