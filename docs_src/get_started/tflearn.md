# tf.contrib.learn 快速入门指南

Tensorflow 的高层次 API（tf.contrib.learn）使得配置、训练和评估许多机器学习模型变得简单起来。在这篇教程中，你将使用 tf.contrib.learn 构建一个[神经网络](https://en.wikipedia.org/wiki/Artificial_neural_network)分类器，然后使用 [Iris 数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set) 进行训练，来根据花的萼片、花瓣几何尺寸来预测花的种类。你将书写代码来完成以下五步：
1.  将包含 Iris 训练集、测试集的 CSV 文件加载到 Tensorflow 的`数据集`
2.  构建一个神经网络分类器 （tf.contrib.learn.DNNClassifier）
3.  使用训练集来训练模型
4.  评估模型的准确率
5.  对新的样本进行分类

注意: 在开始本次教程之前，首先在您的机器上安装Tensorflow

下面是神经网络分类器的全部代码：

```python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# 数据集
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
  # 如果本地没有训练集、测试集，则从网络下载
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
      f.write(raw)

  # 加载数据集
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # 确定所有特征都是实数数据
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # 构建一个三层的深度神经网络（DNN），每层有 10,20,20 个神经元
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="/tmp/iris_model")
  # 定义训练输入
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # 训练模型
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  # 定义测试输入
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # 评估准确率
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # 对两个新的花儿样本进行分类
  def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

  predictions = list(classifier.predict(input_fn=new_samples))

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))

if __name__ == "__main__":
    main()
```

接下来的部分对上面的代码进行详细的解读。

## 加载 Iris CSV 数据到Tensorflow

[Iris数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set) 包含150行数据，其中包括了三种鸢尾花：*Iris setosa*, *Iris virginica*, 和*Iris versicolor* ,每种包含 50 个样本。


![Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor](https://www.tensorflow.org/images/iris_three_species.jpg) **从左到右, [*Iris setosa*](https://commons.wikimedia.org/w/index.php?curid=170298) (by [Radomil](https://commons.wikimedia.org/wiki/User:Radomil), CC BY-SA 3.0), [*Iris versicolor*](https://commons.wikimedia.org/w/index.php?curid=248095) (by [Dlanglois](https://commons.wikimedia.org/wiki/User:Dlanglois), CC BY-SA 3.0),[*Iris virginica*](https://www.flickr.com/photos/33397993@N05/3352169862)
(by [Frank Mayfield](https://www.flickr.com/photos/33397993@N05), CC BY-SA 2.0).**

每行数据包括每个花儿样本的下列数据：[萼片](https://en.wikipedia.org/wiki/Sepal)长度，萼片宽度，[花瓣](https://en.wikipedia.org/wiki/Petal)长度，花瓣宽度和花的种类。花的种类用整数来表示，0 代表 *Iris setosa*, 1 代表 *Iris versicolor*, 2 代表 *Iris virginica*.

萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度 | 种类
:----------- | :---------- | :----------- | :---------- | :-------
5.1          | 3.5         | 1.4          | 0.2         | 0
4.9          | 3.0         | 1.4          | 0.2         | 0
4.7          | 3.2         | 1.3          | 0.2         | 0
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
7.0          | 3.2         | 4.7          | 1.4         | 1
6.4          | 3.2         | 4.5          | 1.5         | 1
6.9          | 3.1         | 4.9          | 1.5         | 1
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
6.5          | 3.0         | 5.2          | 2.0         | 2
6.2          | 3.4         | 5.4          | 2.3         | 2
5.9          | 3.0         | 5.1          | 1.8         | 2

在此次教程中，Iris 数据集被随机拆分成了两个独立的CSV文件：

*   包含 120 个样本的训练集    ([iris_training.csv](http://download.tensorflow.org/data/iris_training.csv))
*   包含 30 个样本的测试集    ([iris_test.csv](http://download.tensorflow.org/data/iris_test.csv)).

我们首先要导入所有需要的模块，并且定义数据下载、保存的位置

```python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
```

然后，如果训练集、测试集没有存储在本地，就从网络上下载它们

```python
if not os.path.exists(IRIS_TRAINING):
  raw = urllib.urlopen(IRIS_TRAINING_URL).read()
  with open(IRIS_TRAINING,'w') as f:
    f.write(raw)

if not os.path.exists(IRIS_TEST):
  raw = urllib.urlopen(IRIS_TEST_URL).read()
  with open(IRIS_TEST,'w') as f:
    f.write(raw)
```

接着，使用 `learn.datasets.base` 中的 [`load_csv_with_header()`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/datasets/base.py) 方法将训练集和验证集加载到`数据集`中。`load_csv_with_header()` 方法需要三个参数：

*   `filename`, 指定 CSV 文件路径
*   `target_dtype`, 指定数据集目标值对应的 [`numpy` 数据类型](http://docs.scipy.org/doc/numpy/user/basics.types.html)
*   `features_dtype`, 指定数据集特征值对应的 [`numpy` 数据类型](http://docs.scipy.org/doc/numpy/user/basics.types.html)


这里，目标值（你训练的模型要预测的值）是花的种类,即从0-2的数字，所以其近似对应的`numpy`数据类型是`np.int`:

```python
# 加载数据集
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
```

tf.contrib.learn 中的`数据集` 是 [named tuples](https://docs.python.org/2/library/collections.html#collections.namedtuple);你可以通过 `data` 和 `target` 来分别获取特征数据和目标值。这里， `training_set.data` 和 `training_set.target` 包含了训练集的特征数据和目标值，相应的， `test_set.data`
和 `test_set.target` 包含了测试集的特征数据和目标值。

接下来,在["使用 Iris 训练数据训练模型,"](#fit-dnnclassifier)章节中你将使用 `training_set.data` 和 `training_set.target` 来训练你的模型，然后在 ["评估模型准确率"](#evaluate-accuracy)章节你会用到 `test_set.data` 和
`test_set.target`。 但你首先要在下一个章节中构建你的模型。

## 构建一个深度神经网络分类器

tf.contrib.learn 提供许多预定义的模型，叫做@{$python/contrib.learn#estimators$`Estimator`s}。你可以直接使用这些 Estimator 在你自己的数据中进行训练和验证，而不用管这些模型内部的具体细节。这里你将配置一个深度神经网络分类器（DNNClassifier）来拟合 Iris 数据。通过使用 tf.contrib.learn, 你可以在短短几行代码内实例化你的 DNNClassifier。

```python
# 确定所有特征都是实数数据
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 构建一个三层的深度神经网络（DNN），每层有 10,20,20 个神经元
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")
```

上面的代码首先定义了模型的特征列，指定了数据集中这些特征的数据类型。因为所有的特征数据都是连续的，所以 `tf.contrib.layers.real_valued_column` 是合适的函数选择来构建特征列。数据集中有四类特征（萼片长度 、萼片宽度、花瓣长度、花瓣宽度 ）,所以相应的维度 `dimension` 应设为4来装载所有的数据。


然后代码使用下列参数构建了一个 `DNNClassifier` 模型： 

*   `feature_columns=feature_columns`. 上面定义的特征列集合
*   `hidden_units=[10, 20, 10]`. 三个[隐层](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw),每层有 10,20,20 个神经元
*   `n_classes=3`. 三个目标类别，分别代表三种鸢尾花种类
*   `model_dir=/tmp/iris_model`. Tensorflow 保存检查点（checkpoint）的文件夹路径。想要了解更多关于 Tensorflow 日志和监控的信息，请查阅 @{$monitors$ tf.contrib.learn 日志和监控的信息}。

## 描述训练时的输入通道（input pipeline）{#train-input}

`tf.contrib.learn` API 使用输入函数来创建为模型生成数据的 TensorFlow 操作。在本例中，考虑到数据集足够小，我们将其存储到@{tf.constant TensorFlow 常量}中。下面的代码创建了最简单输入通道：

```python
# 定义输入
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)

  return x, y
```

## 将 Iris 训练数据装配到 DNNClassifier {#fit-dnnclassifier}

现在你已配置好的你的 DNN `分类`模型了，你可以利用 @{tf.contrib.learn.BaseEstimator.fit$`fit`} 方法，将训练数据装配到分类器上。把 `get_train_inputs` 作为 `input_fn`，然后设置要训练的轮数(这里， 2000)：

```python
# Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)
```

模型的状态保存在 `classifier` 中，这意味着如果你喜欢，你可以反复训练。例如，上面的过程和下面的等价：

```python
classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
```

然而，如果你想要在训练的同时跟踪模型，你可能会使用 Tensorflow 监视器@{tf.contrib.learn.monitors$`monitor`}
来进行日志操作，参见教程@{$monitors$&ldquo;Logging and Monitoring Basics with tf.contrib.learn&rdquo;} 来了解更多。

## 评估模型准确率 {#evaluate-accuracy}

你已经使用 Iris 训练集训练完成你的 `DNNClassifier` 模型了；现在，你可以使用 @{tf.contrib.learn.BaseEstimator.evaluate$`evaluate`} 方法，在 Iris 测试集上检测模型的准确率。和 `fit` 方法类似，`evaluate` 方法需要一个输入函数作为参数来构造其输入通道。`evaluate` 方法返回一个 `dict`，这个字典中包含评估结果。下面的代码将 Iris
测试数据&mdash;`test_set.data` 和 `test_set.target`&mdash;传给 `evaluate` 然后根据结果打印准确率：

```python
# 定义测试输入
def get_test_inputs():
  x = tf.constant(test_set.data)
  y = tf.constant(test_set.target)

  return x, y

# 评估准确率
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                     steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
```

注意：这里 `evaluate` 函数的 `steps` 参数设为 1 很重要。因为@{tf.contrib.learn.Evaluable.evaluate$`evaluate`} 通常直到输入结束时才停止执行。当评估一组输入集合时，这样做非常合适，但是这里使用的常量将永远不会在一轮评估结束时抛出期待的  `OutOfRangeError` 或者 `StopIteration`（译者注：如果 `steps` 大于 1 会评估两轮）。 

当你跑完了整个脚本，他会输出类似于下面的结果：

```
Test Accuracy: 0.966667
```

你的准确率可能有所变化，但应该高于 90% 。这对于一个相当小的数据集来结果说还不错。

## 分类新样本

使用 estimator 的 `predict()` 方法来分类新样本。举个例子， 比如你有下面两个新的花样本： 

萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度
:----------- | :---------- | :----------- | :----------
6.4          | 3.2         | 4.5          | 1.5
5.8          | 3.1         | 5.0          | 1.7

你可以使用 `predict()` 方法来预测它们的种类。`predict` 方法返回一个生成器（generator），它可以很容易地转变成一个列表（list）。下面的代码获取预测结果并输出到屏幕上：

```python
# 对两个新的花儿样本进行分类
def new_samples():
  return np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

predictions = list(classifier.predict(input_fn=new_samples))

print(
    "New Samples, Class Predictions:    {}\n"
    .format(predictions))
```

你的输出结果应该和下面类似：

```
New Samples, Class Predictions:    [1 2]
```

因此，这个模型预测第一个样本是 *Iris versicolor*，第二个样本是 *Iris virginica*。

## 额外资源

*   有关 tf.contrib.learn 进一步的参考资料,请查阅官方API文档@{$python/contrib.learn$ API 文档}。

*   了解更多使用 tf.contrib.learn 构建线性模型的方法,请查看@{$linear$ TensorFlow应用于大规模线性模型}。

*   使用 tf.contrib.learn API 构建你自己的评估器（Estimator）, 查看@{$estimators$在 tf.contrib.learn 中创建评估器}。

*   实验神经网络模型并在浏览器上可视化，请参见 [Deep Playground](http://playground.tensorflow.org/).

*   有关更多神经网络的高级教程，请查看 @{$deep cnn$Convolutional Neural Networks} 和 @{$recurrent$Recurrent Neural Networks}.




