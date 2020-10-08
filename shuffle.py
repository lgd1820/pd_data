import numpy as np
import tensorflow as tf
import sys

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
    train_y = tf.keras.utils.to_categorical(train_y, 2)
    test_y = tf.keras.utils.to_categorical(test_y, 2)

    return train_x, train_y, test_x, test_y

corona = np.load("./npy/Corona.npy")
void = np.load("./npy/Void.npy")

train_x, train_y, test_x, test_y = dataset([corona, void])

np.savez('./shuffle/' + sys.argv[1] + '.npz', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
