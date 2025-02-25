# @Author: Jinyu Zhang
# @Date: 2023/07/27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8


import os
import tensorflow as tf
# from tensorflow.keras import layers
from GPS_Config import *

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


class EA_GPS:
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
        self.dim_external = param.dim_external
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
                    self.encoder(num_items=num_items, num_users=num_users, graph_matrix=laplace_list)
                self.pred = self.decoder(self.graph_ebd_items, self.graph_ebd_users)

            with tf.name_scope('loss'):
                self.loss_mean = loss_calculation(self.target, self.pred)

            with tf.name_scope('optimizer'):
                self.model_op = optimizer(self.loss_mean, self.lr)

    def init_weights(self):
        all_weights = dict()
        # initializer = tf.keras.initializers.glorot_uniform()
        initializer = tf.keras.initializers.random_uniform()
        all_weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.ebd_size]))
        all_weights['item_embedding'] = tf.Variable(initializer([self.num_items, self.ebd_size]))

        # param for EA
        all_weights['mk'] = tf.Variable(initializer(
            [self.num_heads, self.ebd_size * self.dim_coef // self.num_heads, self.dim_external]))
        all_weights['mv'] = tf.Variable(initializer(
            [self.num_heads, self.dim_external, self.ebd_size]))

        # param for prompts
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

    def encoder(self, num_items, num_users, graph_matrix):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # Generate a set of adjacency sub-matrix.
            graph_info = self.unzip_laplace(graph_matrix)

            ego_embeddings = tf.concat([self.all_weights['item_embedding'], self.all_weights['user_embedding']], axis=0)
            all_embeddings = [ego_embeddings]
            for k in range(0, self.num_layers):
                temp_embed = []
                # graph encoder
                for f in range(self.num_folded):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(graph_info[f], ego_embeddings))
                # sum messages of neighbors.
                node_embeddings = tf.concat(temp_embed, 0)
                # external encoder
                external_scores = self.external_attention(node_embeddings, self.all_weights['mk'],
                                                          self.all_weights['mv'], self.dim_coef)
                # add & norm
                node_embeddings = tf.add(external_scores, node_embeddings)
                normed_embeddings = tf.contrib.layers.layer_norm(node_embeddings)

                all_embeddings += [normed_embeddings]

            all_embeddings = tf.stack(all_embeddings, 1)  # layer-wise aggregation
            all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
            # sum the layer aggregation and the normalizer
            graph_ebd_items, graph_ebd_users = tf.split(all_embeddings, [num_items, num_users], 0)

            return graph_ebd_items, graph_ebd_users

    def external_attention(self, graph_node_embeddings, matrix_key, matrix_value, dimension_coefficient):
        element_size = tf.shape(graph_node_embeddings)[0]
        initial_query = tf.keras.layers.Dense(self.ebd_size * dimension_coefficient, activation=None,
                                              kernel_initializer='glorot_uniform')(graph_node_embeddings)
        multi_head_query = tf.reshape(initial_query, (self.num_heads, element_size,
                                                      self.ebd_size * dimension_coefficient // self.num_heads))
        attention_maps = tf.matmul(multi_head_query, matrix_key)  # A = Q · M_k
        normed_ebd = tf.nn.softmax(attention_maps, axis=2)  # ∑ each row of the M_k for normalization
        A_Mv = tf.matmul(normed_ebd, matrix_value)  # score = A · M_v
        merge_heads = tf.reduce_sum(A_Mv, axis=0)
        ebd_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)(merge_heads)

        return ebd_dropout

    def decoder(self, graph_ebd_items, graph_ebd_users):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            item_ebd = tf.nn.embedding_lookup(graph_ebd_items, self.sequence)  # ?, ?, 16
            user_ebd = tf.nn.embedding_lookup(graph_ebd_users, self.user)
            refined_embeds = self.positional_prompt_based_decoder(item_ebd)
            concat_ebd = tf.concat([refined_embeds, user_ebd], axis=1)
            if self.is_train:
                ebd_output = tf.keras.layers.Dropout(rate=self.dropout_rate)(concat_ebd)
            else:
                ebd_output = concat_ebd
            prediction = tf.keras.layers.Dense(self.num_items, activation=None, kernel_initializer='glorot_uniform')(
                ebd_output)
            return prediction

    def positional_prompt_based_decoder(self, item_ebd):
        pos_prompts = tf.nn.embedding_lookup(self.all_weights['pos_embedding'], self.position)  # ?, ?, 16
        composite_embeds = tf.concat([item_ebd, pos_prompts], axis=1)  # ?, ?, 16
        dim_user_0, dim_user_1 = tf.shape(composite_embeds)[0], tf.shape(composite_embeds)[1]
        w_p = self.all_weights['W_p']  # 16, 16
        b_p = self.all_weights['b_p']  # 1, 16
        h_p = self.all_weights['h_p']  # 16, 1
        ebd_form = tf.reshape(composite_embeds, [-1, self.ebd_size])  # ?, 16 linear projection
        mlp_network = tf.nn.relu(tf.matmul(ebd_form, w_p) + b_p)  # [?, 16]
        exp_weights = tf.exp(tf.reshape(tf.matmul(mlp_network, h_p),
                                        [dim_user_0, dim_user_1]))  # [?, sequence_length]
        mask_index = tf.reduce_sum(self.length, axis=1)  # [?, ]
        mask_matrix = tf.sequence_mask(mask_index, maxlen=dim_user_1, dtype=tf.float32)  # [?, sequence_length]
        # soft attention network
        masked_ebd = mask_matrix * exp_weights  # [?, sequence_length]
        exp_sum = tf.pow(tf.reduce_sum(masked_ebd, axis=1, keepdims=True), tf.constant(0.5, tf.float32, [1]))

        prompting_ebd = tf.expand_dims(tf.math.divide(masked_ebd, exp_sum), axis=2)  # [?, sequence_length, 1]

        prompting_ebd = tf.reduce_sum(prompting_ebd * composite_embeds, axis=1)

        return prompting_ebd

    def model_training(self, sess, user, sequence, position, length, target, learning_rate, dropout_rate):

        feed_dict = {self.user: user, self.sequence: sequence, self.position: position, self.length: length,
                     self.target: target,
                     self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = True
        return sess.run([self.loss_mean, self.model_op], feed_dict)

    def model_evaluation(self, sess, user, sequence, position, length, target, learning_rate, dropout_rate):

        feed_dict = {self.user: user, self.sequence: sequence, self.position: position, self.length: length,
                     self.target: target, self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = False
        return sess.run(self.pred, feed_dict)
