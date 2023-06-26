
Model
======

* U-Net: basic three level U-Net
* Context U-Net: three level U-Net with additional Context Modules
* Multihead SPTU-Net: experimental Transformer U-Net

Overview
~~~~~~~~~~

The trained models perform a semantic segmentation of an histological image.

Training and test data
~~~~~~~~~~~~~~~~~~~~~~~~

The training data is the Lizard Dataset which has been patched into 256*256 images according to CoNIC challenge

Model architecture
~~~~~~~~~~~~~~~~~~~~~~

The model is based on `Pytorch <https://pytorch.org/>`_.
On a high level the model can be summarized as follows:
1. 1x convolutional layer
2. 1x rectified linear activation function
3. 1x convolutional layer
4. 1x rectified linear activation function
5. 1x 2D max pooling layer
6. 1x 0.25 dropout layer
7. 1x flatten layer
8. 1x fully connected layer
9. 1x rectified linear activation function
10. 1x 0.25 dropout layer
11. 1x fully connected layer
12. log softmax generating the final output

Evaluation
~~~~~~~~~~~~~

The evaluation was done on a test set from the Lizard DataSet

Hyperparameter selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameters were optimized according to https://arxiv.org/abs/1803.09820

1. ``AdamW optimizer`` was chosen for strong, general performance.
