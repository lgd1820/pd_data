import os
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

cwd = os.getcwd()
data_folder_path = cwd + "\\data\\"

data_folder_list = os.listdir(data_folder_path)
l = []
for data_folder in data_folder_list:
    data_list_path = data_folder_path + data_folder + "\\"
    data_list = os.listdir(data_list_path)
    data_npy = []
    if (not data_folder == "Corona"):
        if (not data_folder == "Void"): continue

    for data in data_list:
        with open(data_list_path + data, "rb") as f:
            byte_data = f.read()
            l = []
            index = 167
            for _ in range(60):
                y_axis = []
                for _ in range(120):
                    x_axis = []
                    for _ in range(60):
                        x_axis.append(byte_data[index])
                        index += 1
                    y_axis.append(x_axis)
                data_npy.append(y_axis)
    data_npy = np.array(data_npy)
    print(data_npy.shape)
    np.save(cwd + "\\npy\\cnn_" + data_folder, data_npy)

'''
for data_folder in data_folder_list:
    data_list_path = data_folder_path + data_folder + "\\"
    data_list = os.listdir(data_list_path)
    data_npy = []
    for data in data_list:
        with open(data_list_path + data, "rb") as f:
            byte_data = f.read()
            l = []
            index = 167
            for _ in range(60):
                y_axis = []
                for _ in range(120):
                    x_axis = []
                    for _ in range(60):
                        x_axis.append(byte_data[index])
                        index += 1
                    y_axis.append(x_axis)
                l.append(y_axis)
            data_npy.append(l)
    data_npy = np.array(data_npy)
    print(data_npy.shape)
    np.save(cwd + "\\npy\\" + data_folder, data_npy)
'''