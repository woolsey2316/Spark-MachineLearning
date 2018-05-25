# Spark-MachineLearning

MNIST is a widely used data set for classification problem. Each MNIST image is a 28 x 28 
grayscale image, using the raw pixel as feature would have a dimension of 748. Spark machine
learning library contains a number of classifiers that can be applied on MNIST data set. The two 
I have used are Logistic Regression and MultiLayer Perception Classifier set and compare their 
performance in terms of accuracy and execution statistics. You need to further explore various 
parameter settings of each classifier.

1. Logistic Regression: the time complexity of Logistic Regression algorithm depends
on the number of training samples, the dimension of the feature vector, and the num-
ber of iterations. Explores the impact of feature vector’s dimension on
execution statistics and prediction accuracy of Spark’s Logistic Regression implemen-
tation. The dimension variations can be created using a dimensionality reduction PCA. 
You should study at least three different dimension values, including one representing raw pixel 
value feature vector (unreduced feature vector).
You can pick the other two dimension values.

2. Multilayer Perceptron Classifier: A Multilayer Perceptron Classifier has many hy-
perparameters that could affect the execution statistics as well as the prediction ac-
curacy.

effect of block size: Build a Multilayer Perceptron classifier with one hidden
layer and a fixed layer size. Choose three blockSize values, with the maximum
being 30(MB). Train the classifier with those values to investigate the effect of
block size on prediction accuracy as well as execution statistics.
