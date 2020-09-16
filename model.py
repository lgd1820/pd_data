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

cwd = os.getcwd()
data_folder_path = cwd + "/npy/"

corona = np.load(data_folder_path + "Corona.npy")
void = np.load(data_folder_path + "Void.npy")

print("corona", corona.shape)
print("void", void.shape)

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
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
           filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
           filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Flatten(),
        #layers.Dropout(0.3),
        #layers.Dense(1000, activation="softmax"),
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

train_x, train_y, test_x, test_y = dataset([corona,void])
seq.fit(
    train_x,
    train_y,
    batch_size=4,
    epochs=20
)

eval_y = seq.evaluate(test_x, test_y, batch_size=5)
print(eval_y)

eval_y = seq.predict(test_x)

np.save("eval_y.npy",eval_y)


y_true = np.array([])
for i in test_y:
    if i[0] > i[1]:
        y_true = np.append(y_true, 0)
    else:
        y_true = np.append(y_true, 1)

y_pred = np.array([])
for i in eval_y:
    if i[0] > i[1]:
        y_pred = np.append(y_pred, 0)
    else:
        y_pred = np.append(y_pred, 1)

print(f1_score(y_true, y_pred, average='macro'))
