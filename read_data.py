# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     read_data
# @Time:         9:24  2023/10/11
import os


def readdtxt(real, name_or_id, datafile, n, base=None):
    """
    :param real:  是否使用real文件夹的数据
    :param name_or_id: real或者simulated数据下的什么文件夹
    :param datafile: 使用第三级文件夹中的data还是ground.truth还是什么
    :param n: 更下一级的第几个文件
    :return:
    """
    if base is None:
        base = '../example-causal-datasets/'
    base_real = base + "real/"
    base_simulated = base + "simulated/"
    if real:
        datasets_real_ = os.listdir(base_real)  # 要读拿个 就base_real + datasets_real_[i]
    else:
        datasets_real_ = os.listdir(base_simulated)  # 要读拿个 就base_real + datasets_real_[i]

    if type(name_or_id) is not int and name_or_id.isdigit():
        datas = base_real + datasets_real_[eval(name_or_id)]
    elif type(name_or_id) is int:
        datas = base_real + datasets_real_[name_or_id]
    else:
        datas = base_real + name_or_id + '/'

    datas_list = os.listdir(datas)

    if type(datafile) is not int and datafile.isdigit():
        data = datas + "/" + datas_list[eval(datafile)]
    elif type(datafile) is int:
        data = datas + "/" + datas_list[datafile]
    else:
        data = datas + "/" + datafile + '/'

    data_list = os.listdir(data)

    return data + data_list[n]


if __name__ == '__main__':
    """
    eg.
    """
    readdtxt(1, "auto-mpg", 'data', 0)
    readdtxt(1, 1, 'data', 2)
