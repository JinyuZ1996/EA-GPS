# @Author: Jinyu Zhang
# @Date: 2023/07/27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp

np.seterr(all='ignore')


class ParamConfig:
    def __init__(self):
        self.code_path = "https://github.com/JinyuZ1996/EA_GPS"

        '''
           block 1: the hyper parameters for model training
        '''
        self.learning_rate = 0.01  # 0.0001 for MovieLens
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.num_epochs = 50   # 300
        self.eval_verbose = 10

        # if you want to implement a fast-running version
        # please consider to set fast_running=Ture
        self.fast_running = False
        self.fast_ratio = 0.2

        '''
            block 2: the hyper parameters for EA_GPS_Module.py
        '''
        self.embedding_size = 16
        self.num_layers = 2
        self.num_folded = self.embedding_size
        self.gpu_index = '0'

        self.num_heads = 2  # Hyper-parameter β
        self.dim_coef = 2 * self.num_heads
        self.dim_external = 128  # Hyper-parameter α
        self.regular_rate = 1e-7
        self.l2_regular_rate = 1e-7

        self.dataset = "Food"  # Food, Book, Movie, MovieLens, Douban
        self.train_path = "../Data/" + self.dataset + "/train_data.txt"
        self.test_path = "../Data/" + self.dataset + "/test_data.txt"
        self.item_path = "../Data/" + self.dataset + "/item_dict.txt"
        self.user_path = "../Data/" + self.dataset + "/user_dict.txt"
        self.check_points = "../check_points/"+self.dataset+".ckpt"


def load_dict(dict_path):
    dict_output = {}
    with open(dict_path, 'r') as file_object:
        elements = file_object.readlines()
    for dict_element in elements:
        dict_element = dict_element.strip().split('\t')
        dict_output[dict_element[1]] = int(dict_element[0])
    return dict_output


def config_input(data_path, item_dict, user_dict):
    with open(data_path, 'r') as data_file:
        data = []
        lines = data_file.readlines()
        for line in lines:
            temp = []
            sequence = []
            position = []
            pos_index = 0
            length = 0
            line_split = line.strip().split('\t')
            sequence.append(user_dict[line_split[0]])
            for index in line_split[1:-1]:
                sequence.append(item_dict[index])
                position.append(pos_index)
                pos_index += 1
                length += 1
            temp.append(sequence)  # user_id & item_id
            temp.append(position)   # positional information for prompt-tuning
            temp.append(length)  # length of the sequence
            temp.append(item_dict[line_split[-1]])  # target_item
            data.append(temp)
    return data


def matrix2list(matrix):
    temp = pd.DataFrame(matrix, columns=['row', 'column'])
    temp.duplicated()
    temp.drop_duplicates(inplace=True)

    return temp.values.tolist()


def trans_matrix_form(data):
    matrix_i2i, matrix_i2u, matrix_u2i, matrix_u2u = [], [], [], []
    matrices_out = []
    for record in data:
        sequence = record[0]
        user = int(sequence[0])  # user_id
        items = [int(i) for i in sequence[1:]]  # user's interacted items
        # construct the sequential relations of items
        for item_index in range(0, len(items) - 1):
            item_temp = items[item_index]
            next_item = items[item_index + 1]
            matrix_i2i.append([item_temp, item_temp])
            matrix_i2i.append([item_temp, next_item])

        # construct the interactive relations between users and items
        for item in items:
            matrix_u2i.append([user, item])
            matrix_i2u.append([item, user])

        matrix_u2u.append([user, user])

    matrices_out.append(np.array(matrix2list(matrix_i2i)))  # [0]
    matrices_out.append(np.array(matrix2list(matrix_i2u)))  # [1]
    matrices_out.append(np.array(matrix2list(matrix_u2i)))  # [2]
    matrices_out.append(np.array(matrix2list(matrix_u2u)))  # [3]

    return matrices_out


def matrix2inverse(array_in, row_index, col_index, matrix_dimension):
    matrix_rows = array_in[:, 0] + row_index
    matrix_columns = array_in[:, 1] + col_index
    matrix_value = [1.] * len(matrix_rows)
    inverse_matrix = sp.coo_matrix((matrix_value, (matrix_rows, matrix_columns)),
                                   shape=(matrix_dimension, matrix_dimension))
    return inverse_matrix


def graph_construction(matrices, item_dict, user_dict):
    graph_matrices = []

    item_size = len(item_dict)  # Note that, the index of the first dictionary's item must be 0
    user_size = len(user_dict)  # if not, here should be "size = len(dict) + 1"
    num_all = item_size + user_size

    # Form a complete graph (including the sequence relationship
    # between items and the interaction relationship between user and items)
    matrix_i2i = matrix2inverse(matrices[0], row_index=0, col_index=0, matrix_dimension=num_all)
    matrix_i2u = matrix2inverse(matrices[1], row_index=0, col_index=item_size, matrix_dimension=num_all)
    matrix_u2i = matrix2inverse(matrices[2], row_index=item_size, col_index=0, matrix_dimension=num_all)
    matrix_u2u = matrix2inverse(matrices[3], row_index=item_size, col_index=item_size, matrix_dimension=num_all)

    graph_matrices.append(matrix_i2i)
    graph_matrices.append(matrix_i2u)
    graph_matrices.append(matrix_u2i)
    graph_matrices.append(matrix_u2u)

    laplace_list = [adj.tocoo() for adj in graph_matrices]

    return sum(laplace_list)


def load_batches(batch, padding_num):
    user, sequence, position, length, target = [], [], [], [], []
    for data_index in batch:
        length.append(data_index[2])
    max_length = max(length)

    i = 0
    for data_index in range(len(batch)):
        user.append(batch[data_index][0][0])
        sequence.append(batch[data_index][0][1:] + [padding_num] * (max_length - length[i]))
        position.append(batch[data_index][1][0:] + [padding_num] * (max_length - length[i]))
        target.append(batch[data_index][3])
        i += 1

    return np.array(user), np.array(sequence), np.array(position), np.array(length).reshape(len(length), 1), np.array(target)


def generate_batches(input_data, batch_size, padding_num, is_train):
    user_all, sequence_all, position_all, length_all, target_all = [], [], [], [], []
    num_batches = int(len(input_data) / batch_size)

    if is_train is True:
        random.shuffle(input_data)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        batch = input_data[start_index:start_index + batch_size]
        user, sequence, position, length, target = load_batches(batch=batch, padding_num=padding_num)

        user_all.append(user)
        sequence_all.append(sequence)
        position_all.append(position)
        length_all.append(length)
        target_all.append(target)

    return list((user_all, sequence_all, position_all, length_all, target_all, num_batches))
