# MNIST_Recognition
对数据集MNIST的识别和特征可视化。

1.使用机器学习方法识别

（1）Logistic回归

对于多分类问题，使用One vs Rest模型进行判别，最终准确率约90.39%。

（2）Softmax回归

实现Softmax函数及相应的损失函数，最终准确率约92.05%。

2.对图片的特征进行可视化

（1）通过CNN降维至128维

（2）先PCA降维（降到30、50、80等）后再进行T-SNE降维至二维，展示结果。

（3）使用AutoEncoder展示3D可视化特征。