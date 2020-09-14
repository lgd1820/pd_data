'''
작성일 : 2020-09-14
작성자 : 이권동
코드 개요 : 데이터 전처리를 담당하는 코드
'''
import os
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

cwd = os.getcwd()
data_folder_path = cwd + "./data/"

data_folder_list = os.listdir(data_folder_path)
l = []

# data 폴더 안에 있는 데이터를 전처리하는 반복문
# 데이터는 헤더 127 byte, 60 * 128 * 60 bytes 형태로 이루어짐
# 현재 테스트 하는 데이터는 corona, void
for data_folder in data_folder_list:
    data_list_path = data_folder_path + data_folder + "/"
    data_list = os.listdir(data_list_path)
    data_npy = []
    for data in data_list:
        with open(data_list_path + data, "rb") as f:
            byte_data = f.read()
            l = []
            index = 167
            for _ in range(60):
                y_axis = []
                for _ in range(128):
                    x_axis = []
                    for _ in range(60):
                        x_axis.append(byte_data[index])
                        index += 1
                    y_axis.append(x_axis)
                l.append(y_axis)
            data_npy.append(l)
    data_npy = np.array(data_npy)
    print(data_npy.shape)
    np.save(cwd + "/npy/" + data_folder, data_npy)
