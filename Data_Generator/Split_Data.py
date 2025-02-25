import collections
import scipy.sparse as sp
import numpy as np
import pandas as pd
import os
import random
import itertools


def load_dict(dict_path):
    itemdict = {}
    with open(dict_path, 'r') as file_object:
        items = file_object.readlines()
    for item in items:
        item = item.strip().split('\t')
        itemdict[item[1]] = int(item[0])
    return itemdict


def get_data(data_path, data_Movie, data_Book, dict_A):
    movie_object = open(data_Movie, "w")
    book_object = open(data_Book, "w")
    with open(data_path, 'r') as file_object:
        # mixed_data = []
        lines = file_object.readlines()
        for line in lines:
            # temp_sequence = []
            line = line.strip().split('\t')
            sequence_A = ""
            sequence_B = ""
            user = line[0]  # 每行第一个是uid
            # sequence_all.append(dict_U[user])  # 现在混合seq第一个位置上拼上uid
            sequence_A += str(user)
            sequence_A += "\t"
            sequence_B += str(user)
            sequence_B += "\t"
            for item in line[1:]:
                item_info = item.split('|')
                item_id = item_info[0]
                # item_id = item_info
                if item_id in dict_A:
                    # sequence_all.append(dict_A[item_id])
                    sequence_A += str(item_id)
                    sequence_A += "\t"
                else:
                    # sequence_all.append(dict_B[item_id] + len(dict_A))  # 为了区分序列中的E与V物品，len(itemE)=8367；
                    sequence_B += str(item_id)
                    sequence_B += "\t"
            # sequence_A.append("\n")
            # sequence_B.append("\n")
            movie_object.write(sequence_A + "\n")
            book_object.write(sequence_B + "\n")
            # temp_sequence.append(sequence_all)  # [0]
            # mixed_data.append(temp_sequence)
    return


dict_A = load_dict("../Data_Generator/item_dict.txt")
get_data("../Data_Generator/train_data.txt", "../Data_Generator/Video.txt", "../Data_Generator/Novel.txt", dict_A)
