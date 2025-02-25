# @Author: Jinyu Zhang
# @Date: 2023/07/27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

from GPS_Config import *
import os

param = ParamConfig()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index


def train_module(sess, module, batches_train):
    user_all, sequence_all, position_all, length_all, target_all, train_batch_num = (batches_train[0], batches_train[1],
                                                                                     batches_train[2], batches_train[3],
                                                                                     batches_train[4], batches_train[5])

    shuffled_batch_indexes = np.random.permutation(train_batch_num)
    loss_sum = 0

    for batch_index in shuffled_batch_indexes:
        user = user_all[batch_index]
        sequence = sequence_all[batch_index]
        position = position_all[batch_index]
        length = length_all[batch_index]
        target = target_all[batch_index]

        batch_loss, _ = module.model_training(sess=sess, user=user, sequence=sequence, position=position, length=length,
                                              target=target,
                                              learning_rate=param.learning_rate, dropout_rate=param.dropout_rate)
        loss_sum += batch_loss

    avg_loss = loss_sum / train_batch_num
    return avg_loss


def evaluate_module(sess, module, batches_test, eval_length):
    user_all, sequence_all, position_all, length_all, target_all, test_batch_num = (
    batches_test[0], batches_test[1], batches_test[2],
    batches_test[3], batches_test[4], batches_test[5])

    return evaluate_ratings(sess=sess, module=module, user_all=user_all, sequence_all=sequence_all,
                            position_all=position_all,
                            length_all=length_all, target_all=target_all, num_batches=test_batch_num,
                            eval_length=eval_length)


def evaluate_ratings(sess, module, user_all, sequence_all, position_all, length_all, target_all, num_batches,
                     eval_length):
    rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20 = 0, 0, 0, 0, 0, 0

    for batch_index in range(num_batches):
        test_user = user_all[batch_index]
        test_sequence = sequence_all[batch_index]
        test_position = position_all[batch_index]
        test_length = length_all[batch_index]
        test_target = target_all[batch_index]

        prediction = module.model_evaluation(sess=sess, user=test_user, sequence=test_sequence, position=test_position,
                                             length=test_length, target=test_target, learning_rate=param.learning_rate,
                                             dropout_rate=0)
        recall, mrr = eval_metrics(prediction, test_target, [5, 10, 20])
        rc_5 += recall[0]
        rc_10 += recall[1]
        rc_20 += recall[2]
        mrr_5 += mrr[0]
        mrr_10 += mrr[1]
        mrr_20 += mrr[2]

    return [rc_5 / eval_length, rc_10 / eval_length, rc_20 / eval_length, mrr_5 / eval_length, mrr_10 / eval_length,
            mrr_20 / eval_length]


def eval_metrics(pred_list, target_list, options):
    recall, mrr = [], []
    pred_list = pred_list.argsort()
    for k in options:
        recall.append(0)
        mrr.append(0)
        temp_list = pred_list[:, -k:]
        search_index = 0
        while search_index < len(target_list):
            pos = np.argwhere(temp_list[search_index] == target_list[search_index])
            if len(pos) > 0:
                recall[-1] += 1
                mrr[-1] += 1 / (k - pos[0][0])
            else:
                recall[-1] += 0
                mrr[-1] += 0
            search_index += 1
    return recall, mrr
