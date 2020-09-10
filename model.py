from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os

cwd = os.getcwd()
data_folder_path = cwd + "/npy/"

corona = np.load(data_folder_path + "Corona.npy")
void = np.load(data_folder_path + "Void.npy")

print("corona", corona.shape)
print("void", void.shape)

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
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.FalseNegatives(name='fn'),])

seq.summary()

train_x, train_y, test_x, test_y = dataset([corona,void])

seq.fit(
    train_x,
    train_y,
    batch_size=4,
    epochs=10
)

eval_y = seq.evaluate(test_x, test_y, batch_size=5)
print(eval_y)

# eval_y = seq.predict(test_x)
# print(train_x.shape, test_x.shape)
# print(eval_y)
