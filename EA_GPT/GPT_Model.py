# @Author: Jinyu Zhang
# @Date: 2023/07/27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8


import os
import tensorflow as tf
from GPT_Config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2023
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
param = ParamConfig()


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def get_inputs():
    user = tf.placeholder(dtype=tf.int32, shape=[None, ], name='user')
    sequence = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence')
    position = tf.placeholder(dtype=tf.int32, shape=[None, None], name="position")
    length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='length')
    target = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
    return user, sequence, position, length, target, learning_rate, dropout_rate


def loss_calculation(target, pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=pred)
    loss_mean = tf.reduce_mean(loss, name='loss_mean')
    return loss_mean


def optimizer(loss, learning_rate):
    basic_op = tf.train.AdamOptimizer(learning_rate)
    gradients = basic_op.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                        grad is not None]
    model_op = basic_op.apply_gradients(capped_gradients)
    return model_op


class EA_GPT:
    def __init__(self, num_items, num_users, laplace_list):
        os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items = num_items
        self.num_users = num_users
        self.laplace_list = laplace_list

        self.num_heads = param.num_heads
        self.dim_coef = param.dim_coef
        self.batch_size = param.batch_size
        self.regular_rate = param.regular_rate

        self.ebd_size = param.embedding_size
        self.num_layers = param.num_layers
        self.num_folded = param.num_folded
        self.is_train = True

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.user, self.sequence, self.position, self.length, self.target, self.lr, self.dropout_rate = get_inputs()

            with tf.name_scope('encoder_decoder'):
                self.all_weights = self.init_weights()
                self.graph_ebd_items, self.graph_ebd_users = \
                    self.parallel_encoder(num_items=num_items, num_users=num_users, graph_matrix=laplace_list)
                self.pred = self.decoder(self.graph_ebd_items, self.graph_ebd_users)

            with tf.name_scope('loss'):
                self.loss_mean = loss_calculation(self.target, self.pred)

            with tf.name_scope('optimizer'):
                self.model_op = optimizer(self.loss_mean, self.lr)

    def init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.ebd_size]))
        all_weights['item_embedding'] = tf.Variable(initializer([self.num_items, self.ebd_size]))
        all_weights['pos_embedding'] = tf.Variable(initializer([self.num_items, self.ebd_size]))
        all_weights['W_p'] = tf.Variable(initializer([self.ebd_size, self.ebd_size]), dtype=tf.float32)
        all_weights['b_p'] = tf.Variable(initializer([1, self.ebd_size]), dtype=tf.float32)
        all_weights['h_p'] = tf.Variable(tf.ones([self.ebd_size, 1]), dtype=tf.float32)
        return all_weights

    def unzip_laplace(self, X):
        unzip_info = []
        fold_len = (X.shape[0]) // self.num_folded
        for i_fold in range(self.num_folded):
            start = i_fold * fold_len
            if i_fold == self.num_folded - 1:
                end = X.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            unzip_info.append(_convert_sp_mat_to_sp_tensor(X[start:end]))
        return unzip_info

    def parallel_encoder(self, num_items, num_users, graph_matrix):
        # Generate a set of adjacency sub-matrix.
        graph_info = self.unzip_laplace(graph_matrix)

        ego_embeddings = tf.concat([self.all_weights['item_embedding'], self.all_weights['user_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.num_layers):
            temp_embed = []
            for f in range(self.num_folded):
                temp_embed.append(tf.sparse_tensor_dense_matmul(graph_info[f], ego_embeddings))
            # sum messages of neighbors.
            node_embeddings = tf.concat(temp_embed, 0)
            node_embeddings = self.external_encoder(ebd_in=node_embeddings)
            all_embeddings += [node_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)  # layer-wise aggregation
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        # sum the layer aggregation and the normalizer
        graph_ebd_items, graph_ebd_users = tf.split(all_embeddings, [num_items, num_users], 0)

        return graph_ebd_items, graph_ebd_users

    def external_encoder(self, ebd_in):
        with tf.variable_scope('external_encoder', reuse=tf.AUTO_REUSE):
            dim_nodes = tf.shape(ebd_in)[0]
            basic_mapping = tf.layers.dense(ebd_in, self.ebd_size * self.dim_coef, activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # [?, 64]
            query_multi_heads = tf.reshape(basic_mapping, shape=(dim_nodes, self.num_heads,
                                                                 self.ebd_size * self.dim_coef // self.num_heads))  # [?, 2, 32]
            key_mapping = tf.layers.dense(query_multi_heads, self.batch_size // self.dim_coef,
                                          activation=tf.nn.leaky_relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # [?, 2, 64]
            ebd_norm_1 = tf.math.l2_normalize(key_mapping, axis=2)
            value_cal = tf.layers.dense(ebd_norm_1, self.ebd_size, activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # [?, 2, 64]
            ebd_sum = tf.reduce_sum(value_cal, axis=1)  # [?, 64]
            ebd_norm_2 = tf.math.l2_normalize(ebd_sum, axis=1)
            # ebd_drop_2 = tf.nn.dropout(ebd_norm_2, 1 - self.dropout_rate)
            ebd_drop_2 = tf.layers.dropout(ebd_norm_2, rate=self.dropout_rate,
                                           training=tf.convert_to_tensor(self.is_train))
        return ebd_drop_2

    def decoder(self, graph_ebd_items, graph_ebd_users):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            item_ebd = tf.nn.embedding_lookup(graph_ebd_items, self.sequence)
            pos_prompts = tf.nn.embedding_lookup(self.all_weights['pos_embedding'], self.position)
            item_ebd = tf.concat([item_ebd, pos_prompts], axis=1)
            dim_user_0, dim_user_1 = tf.shape(item_ebd)[0], tf.shape(item_ebd)[1]
            ebd_form = tf.reshape(item_ebd, [-1, self.ebd_size])
            mlp_network = tf.nn.relu(tf.matmul(ebd_form, self.all_weights['W_p']) + self.all_weights['b_p'])  # [?, 16]
            exp_weights = tf.exp(tf.reshape(tf.matmul(mlp_network, self.all_weights['h_p']), [dim_user_0, dim_user_1]))  # [?, ?]
            mask_index = tf.reduce_sum(self.length, axis=1)  # [?, ]
            mask_matrix = tf.sequence_mask(mask_index, maxlen=dim_user_1, dtype=tf.float32)  # [?, ?]
            masked_ebd = mask_matrix * exp_weights  # [?, ?]
            exp_sum = tf.reduce_sum(masked_ebd, axis=1, keepdims=True)  # [?, 1]
            exp_sum = tf.pow(exp_sum, tf.constant(0.5, tf.float32, [1]))  # [?, 1]
            prompting_ebd = tf.expand_dims(tf.div(masked_ebd, exp_sum), axis=2)  # [?, ?, 1]
            ebd_scored = tf.reduce_sum(prompting_ebd * item_ebd, axis=1)
            user_ebd = tf.nn.embedding_lookup(graph_ebd_users, self.user)
            concat_ebd = tf.concat([ebd_scored, user_ebd], axis=1)
            ebd_output = tf.layers.dropout(concat_ebd, rate=self.dropout_rate,
                                           training=tf.convert_to_tensor(self.is_train))
            prediction = tf.layers.dense(ebd_output, self.num_items, activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return prediction

    def model_training(self, sess, user, sequence, position, length, target, learning_rate, dropout_rate):

        feed_dict = {self.user: user, self.sequence: sequence, self.position: position, self.length: length, self.target: target,
                     self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = True
        return sess.run([self.loss_mean, self.model_op], feed_dict)

    def model_evaluation(self, sess, user, sequence, position, length, target, learning_rate, dropout_rate):

        feed_dict = {self.user: user, self.sequence: sequence, self.position:position, self.length: length,
                     self.target: target, self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = False
        return sess.run(self.pred, feed_dict)
