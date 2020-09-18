'''
작성일 : 2020-09-14
작성자 : 이권동
코드 개요 : ConvLSTM2D를 이용해 cnn으로 된 시계열 데이터를 학습하는 코드
'''
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os

'''
    함수 개요 :
        학습 데이터와 테스트 데이터를 split 하는 함수
    매개변수 :
        npy_list = npy 데이터의 리스트
        num = 학습 데이터로 사용할 개수
'''
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

# 학습 네트워크
# 이부분은 계속 수정 중
seq = keras.Sequential(
    [
        keras.Input(
            shape=(60, 128, 60, 1)
        ),
        layers.Conv3D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPool3D(pool_size=(2, 2, 1), padding='same', data_format='channels_first'),
        layers.Conv3D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPool3D(pool_size=(2, 2, 1), padding='same', data_format='channels_first'),
        layers.Conv3D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPool3D(pool_size=(2, 2, 1), padding='same', data_format='channels_first'),
        layers.Conv3D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPool3D(pool_size=(2, 2, 1), padding='same', data_format='channels_first'),

        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(1000, activation="relu"),
        layers.Dropout(0.3),
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

seq.fit(
    train_x,
    train_y,
    batch_size=4,
    epochs=20
)

eval_y = seq.evaluate(test_x, test_y, batch_size=5)
print(eval_y)

line = str(eval_y[0]) + "," + str(eval_y[1]) + "," + str(sum(eval_y[6])/2) + "\n"
#line = str(train_val[0]) + "," + str(train_val[1]) + "," + str(sum(train_val[6])/2) + \

with open('acc', "a") as f:
    f.write(line)

