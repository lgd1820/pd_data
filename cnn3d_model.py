from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os


def data_processing(rate=0.8):
    cwd = os.getcwd()
    data_folder_path = cwd + "\\npy\\"

    corona = np.load(data_folder_path + "Corona.npy")
    void = np.load(data_folder_path + "Void.npy")   

    print("corona", corona.shape)
    print("void", void.shape)

    if len(corona) > len(void):
        index = int(len(void) * rate)
    else:
        index = int(len(void) * rate)
    
    corona_shuffle_indices = np.random.permutation(np.arange(len(corona)))
    void_shuffle_indices = np.random.permutation(np.arange(len(void)))

    corona = corona[corona_shuffle_indices]
    void = void[void_shuffle_indices]

    train_x = np.concatenate((corona[:index], void[:index]))
    test_x = np.concatenate((corona[index:], void[index:]))
    train_y = np.concatenate(([0 for _ in range(index)], [1 for _ in range(index)]))
    test_y = np.concatenate(([0 for _ in range(len(corona[index:]))], [1 for _ in range(len(void[index:]))]))

    train_x = train_x.reshape(-1, 60, 120, 60, 1).astype('float32')
    test_x = test_x.reshape(-1, 60, 120, 60, 1).astype('float32')
    train_y = tf.keras.utils.to_categorical(train_y, 2)
    test_y = tf.keras.utils.to_categorical(test_y, 2)

    return train_x, train_y, test_x, test_y

def set_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(8, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(60, 120, 60, 1)))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same'))
    model.add(tf.keras.layers.Conv3D(16, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same'))
    model.add(tf.keras.layers.Conv3D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same'))
    #model.add(tf.keras.layers.Conv3D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
        metrics=['accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.FalsePositives(name='false_positives'),
            tf.keras.metrics.FalseNegatives(name='false_negatives')])
    
    model.summary()
    return model

train_x, train_y, test_x, test_y = data_processing()

model = set_model()

model.fit(train_x, train_y, epochs=20, batch_size=8)

eval_y = model.evaluate(test_x, test_y, batch_size=8)
print(train_x.shape, test_x.shape)
print(eval_y)