
# Deep learning

Deep learning refers to highly multi-layer neural networks that have lots of parameters. Training them can be slow, so be prepared to wait if you have a low-end PC. 

Let's have a look at some popular frameworks for deep learning. The most popular is [tensorflow](https://www.tensorflow.org/), that allows one to create complex computing graphs in Python, while the actual heavy lifting is done by underlying C++ implementations. While tensorflow itself is really powerful, we'll be having a look at [keras](https://keras.io/), an abstraction on top of tensorflow that lets you define neural network in an easy manner.

If you're new to Jupyter notebooks, you can run a cell by clicking on it and pressing `ctrl` + `enter`. The variables, functions etc in previous cells will remain in memory, so you can refer to them later in other cells.

This exercise has some ready-made code and you should implement the parts where is says #TODO.

**First things first, download the [HASYv2](https://zenodo.org/record/259444#.WcZjIZ8xDCI) dataset into the same folder as this notebook, and extract it.**


```python
# Python2 compatibility
from __future__ import print_function

import numpy as np
import pandas as pd
```


Implement a function that reads the HASYv2 dataset from the given folder. It should return a tuple containing two numpy arrays: the data as a `(68233, 32, 32)` tensor, and a size `168233` vector containing the labels ("symbol_id") as integers.
Hint. Use scipy's imread to read the `.png` images as black-and-white. Each image should be a 32x32 matrix.


```python
from scipy.misc import imread

# TODO 
def read_data(folder):
    labels = pd.read_csv(folder + "hasy-data-labels.csv")
    X = list()
    for idx, path in enumerate(labels["path"]):
        if idx % 10000 == 0:
            print("at",idx)
            
        img = imread(folder + path, mode="L")
        X.append(img)
    return (np.stack(X), np.array(labels["symbol_id"]))
    

X, y = read_data("HASYv2/")

print(X.shape, y.shape)
```

    at 0
    at 10000
    at 20000
    at 30000
    at 40000
    at 50000
    at 60000
    at 70000
    at 80000
    at 90000
    at 100000
    at 110000
    at 120000
    at 130000
    at 140000
    at 150000
    at 160000
    (168233, 32, 32) (168233,)


Overfitting is when we fit the model to work well on our training data, but . Fitting a model to predict the training set perfectly is in most cases trivial, and is a simple exercise in [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization). In machine learning, however, we are interested in _generalization error_, or in other words, how well our model works on previously unseen data. Therefore, we want to evaluate the performance of our model on data that it has not yet seen: this is a way to approximate how well it generalizes. 

As a reminder from last week, tuning the hyperparameters of your model or choosing your model based on the test data **overfits to the test data**. We don't want this, so we use a so-called _validation set_ to tune our model.

To do this, split the data into training, validation and test sets. Be sure to shuffle it first, otherwise your validation and test sets will not contain most labels. Your function should return a tuple containing the training, validation and test data, i.e `(X_train, y_train, X_val, y_val, X_test, y_test)`. You can use e.g the proportions `0.8, 0.1, 0.1`.


```python
from sklearn.utils import shuffle

# TODO
def split_data(X, y):
    X, y = shuffle(X, y, random_state=0)
    
    train_amt = int(0.8 * X.shape[0])
    X_train = X[:train_amt, :, :]
    y_train = y[:train_amt]
    
    val_amt = int(0.1 * X.shape[0])
    X_val = X[train_amt:train_amt + val_amt, :, :]
    y_val = y[train_amt:train_amt + val_amt]
    
    X_test = X[train_amt+val_amt:, :, :]
    y_test = y[train_amt+val_amt:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test 
    
    
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

print(X_train.shape, y_train.shape)
```

    (134586, 32, 32) (134586,)


Currently our labels are single integers that represent the class. However, for neural networks it's common to switch them to a so-called "one-hot" encoding, where each label is represented by a vector of length number of classes that has a one at the position  zeros everywhere else. 

For example, if there were 7 classes, we could represent that class 5 as `0000100`. In the case of the HASYv2 dataset, there's 369 unique classes, **but because of computational complexity, only pick the data from the first 100 classes** so each label should be a length 100 vector with a single one.


```python
# TODO convert labels to one-hot encoding here
from keras.utils import to_categorical

unique = np.unique(y)

def get_id(x):
    return np.where(unique == x)[0][0]

ids = np.vectorize(get_id)(y)
y_train, y_val, y_test = map(np.vectorize(get_id), [y_train, y_val, y_test])


print(y_train.shape)
```

    (134586,)



```python
# Only take the first 100 classes
inds = np.where(np.logical_and(y_train>=0, y_train<=100))

print(inds[0].shape)

X_train = X_train[inds[0], :, :]
y_train = y_train[inds[0]]

inds = np.where(np.logical_and(y_test>=0, y_test<=100))

X_test = X_test[inds[0], :, :]
y_test = y_test[inds[0]]



y_train, y_val, y_test = map(to_categorical, [y_train, y_val, y_test])
print(y_train.shape, X_train.shape)
```

    (36912,)
    (36912, 101) (36912, 32, 32)


Next let's create a simple linear model using Keras to get ourselves familiar with it. 


```python
from keras.models import Sequential
from keras.layers import Dense, Flatten

# TODO
def create_linear_model():
    model = Sequential()
    
    model.add(Flatten(input_shape=(32, 32)))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    
    return model

model = create_linear_model()
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_2 (Flatten)          (None, 1024)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 101)               103525    
    =================================================================
    Total params: 103,525
    Trainable params: 103,525
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, batch_size=64)
```

    Epoch 1/3
    36912/36912 [==============================] - 2s - loss: 15.7673 - acc: 0.0218     
    Epoch 2/3
    36912/36912 [==============================] - 2s - loss: 15.7648 - acc: 0.0219     
    Epoch 3/3
    36912/36912 [==============================] - 2s - loss: 15.7648 - acc: 0.0219     





    <keras.callbacks.History at 0x7f2ca0f3ca58>




```python
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.backend import clear_session


# TODO
def create_convolutional_model():
    model = Sequential()
    
    model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 1)))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(128, (2, 2), activation="relu"))
    model.add(MaxPooling2D((4, 4)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(y_train.shape[1], activation="softmax"))
    
    return model

clear_session()

model = create_convolutional_model()
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 30, 30, 128)       1280      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 6, 6, 128)         65664     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 1, 1, 128)         0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 1, 1, 128)         512       
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 64)                256       
    _________________________________________________________________
    dense_2 (Dense)              (None, 101)               6565      
    =================================================================
    Total params: 82,533
    Trainable params: 82,149
    Non-trainable params: 384
    _________________________________________________________________



```python
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit(X_train[:, :, :, np.newaxis], y_train, epochs=3, batch_size=128)
```

    Epoch 1/3
    36912/36912 [==============================] - 56s - loss: 2.0671 - acc: 0.5155    
    Epoch 2/3
    36912/36912 [==============================] - 56s - loss: 1.2086 - acc: 0.6586    
    Epoch 3/3
    36912/36912 [==============================] - 56s - loss: 1.0098 - acc: 0.6985    





    <keras.callbacks.History at 0x7f2ca0f3c6a0>




```python
model.evaluate(X_test[:, :, :, np.newaxis], y_test)
```

    4576/4591 [============================>.] - ETA: 0s




    [1.7253065092115458, 0.5231975605222452]




```python

```
