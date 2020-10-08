'''
작성일 : 2020-09-14
작성자 : 이권동
코드 개요 : ConvLSTM2D를 이용해 cnn으로 된 시계열 데이터를 학습하는 코드
'''
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import sys

# 학습 네트워크
# 이부분은 계속 수정 중
seq = keras.Sequential(
    [
        keras.Input(
            shape=(60, 128, 60, 1)
        ), 
        layers.ConvLSTM2D(
            filters=16, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.MaxPool3D(pool_size=(2, 2, 1), padding='same', data_format='channels_first'),
        layers.ConvLSTM2D(
            filters=16, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.MaxPool3D(pool_size=(2, 2, 1), padding='same', data_format='channels_first'),
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(1000, activation="softmax"),
        layers.Dropout(0.1),
        layers.Dense(2, activation="softmax")
    ]
)

#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

seq.compile(loss="categorical_crossentropy", optimizer='adam',
    metrics=['accuracy',
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tfa.metrics.F1Score(num_classes=2, average=None)])

seq.summary()

dataset = np.load('./shuffle/' + sys.argv[1] + '.npz')

train_x, train_y, test_x, test_y = dataset["train_x"], dataset["train_y"], dataset["test_x"], dataset["test_y"]
train_x = train_x.reshape(-1, 60, 128, 60, 1).astype('float32')
test_x = test_x.reshape(-1, 60, 128, 60, 1).astype('float32')

train_val = seq.fit(
    train_x,
    train_y,
    batch_size=2,
    epochs=1
)

print(train_val)

eval_y = seq.evaluate(test_x, test_y, batch_size=5)

line = str(eval_y[0]) + "," + str(eval_y[1]) + "," + str(sum(eval_y[6])/2) + "\n"
#line = str(train_val[0]) + "," + str(train_val[1]) + "," + str(sum(train_val[6])/2) + \

with open('acc', "a") as f:
    f.write(line)


