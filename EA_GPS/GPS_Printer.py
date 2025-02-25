# @Author: Jinyu Zhang
# @Date: 2023/07/27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import logging


def print_train(epoch, loss, time_consumption):
    print('Epoch {} - Training Loss: {:.5f} - Training time: {:.3}'.format(epoch, loss, time_consumption))
    logging.info('Epoch {} - Training Loss: {:.5f} - Training time: {:.3}'.format(epoch, loss, time_consumption))


def print_evaluation(epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20, test_consumption):
    print("Evaluation at Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f" %
          (epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20))
    logging.info("Evaluation at Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, "
                 "MRR20 = %.4f" % (epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20))

    print("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, test_consumption))
    logging.info("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, test_consumption))