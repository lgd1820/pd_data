from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import f1_score
import numpy as np
import os

def dataset(npy_list, num=80):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    class_num = 0
    for npy in npy_list:
        shuffle_indices = np.random.permutation(np.arange(len(npy)))
        npy = npy[shuffle_indices]
        y = np.array([class_num for _ in range(len(npy))])
        if class_num == 0:
            train_x = npy[:num]
            train_y = y[:num]
            test_x = npy[num:]
            test_y = y[num:]
        else:
            train_x = np.concatenate((train_x,npy[:num]))
            train_y = np.concatenate((train_y,y[:num]))
            test_x = np.concatenate((test_x,npy[num:]))
            test_y = np.concatenate((test_y,y[num:]))
        class_num += 1

    print(train_x.shape)
    print(test_x.shape)
    train_x = train_x.reshape(-1, 60, 128, 60, 1).astype('float32')
    test_x = test_x.reshape(-1, 60, 128, 60, 1).astype('float32')
    #train_x = train_x / 255.0
    #test_x = test_x / 255.0
    train_y = tf.keras.utils.to_categorical(train_y, 2)
    test_y = tf.keras.utils.to_categorical(test_y, 2)

    return train_x, train_y, test_x, test_y

corona = np.load("./npy/Corona.npy")
void = np.load("./npy/Void.npy")

train_x, train_y, test_x, test_y = dataset([corona,void])

eval_y = np.load("eval_y.npy")

print(eval_y)
print(eval_y.shape)

y_pred = np.array([])
for i in eval_y:
    if i[0] > i[1]:
        y_pred = np.append(y_pred, 0)
    else:
        y_pred = np.append(y_pred, 1)

print(y_pred)

y_true = np.array([])
for i in test_y:
    if i[0] > i[1]:
        y_true = np.append(y_true,0)
    else:
        y_true = np.append(y_true,1)

y_pred = y_pred.reshape(-1, 1)
y_true = y_true.reshape(-1, 1)

#t, p = [1,1,0,1,1,0], [1,0,1,1,0,1]
#y_true = tf.constant(t, shape=(6,1))

print(y_pred.shape, y_true.shape)

print(f1_score(y_true, y_pred, average='macro'))
#f1 = tfa.metrics.F1Score(num_classes=2, average=None)
#f1.update_state(y_true, y_pred)
#print('F1 Score is: ', f1.result().numpy())
