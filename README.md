# Implementation of Convolutional Neural Network for sentence classification in Tensorflow

This repository contains the Python script as well as the source dataset. The dataset contains 10,662 movie review sentences from RottenTomatoes, of which 5331 are positive and 5331 are negative. The dataset has a vocabulary size of 18,765.

Data Preprocessing is implemented in data_preprocessing_helper module where the data is cleaned and each sentence is padded to maximum sentence length of 59. A vocabulary index is built and each word is mapped to an integer between 0 and 18,765.

In the defined CNN, first layers embeds words into low-dimensional vectors. The next layer performs convolutions over the embedded word vectors using multiple filter sizes. Next, we max-pool the result of the convolutional layer into a long feature vector, add dropout regularization, and classify the result using a softmax layer.

Cross-entropy loss function is being used in our defined model. Adaptive Moment Estimation optimizer is used to optimize the network's loss function.

Results can be visualized by running the following command:

tensorboard --logdir /PATH_TO_CODE/runs/1449760558/summaries/
