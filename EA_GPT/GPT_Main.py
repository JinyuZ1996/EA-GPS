# @Author: Jinyu Zhang
# @Date: 2023/07/27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

from time import time
from EA_GPT.GPT_Printer import *
from EA_GPT.GPT_Train import *
from EA_GPT.GPT_Model import *

param = ParamConfig()

if __name__ == '__main__':

    item_dict = load_dict(dict_path=param.item_path)
    user_dict = load_dict(dict_path=param.user_path)
    print("Dictionaries initialized. Loading data...")

    train_data = config_input(data_path=param.train_path, item_dict=item_dict, user_dict=user_dict)
    test_data = config_input(data_path=param.test_path, item_dict=item_dict, user_dict=user_dict)
    if param.fast_running:
        train_data = train_data[:int(param.fast_ratio*len(train_data))]
        print("Data initialized (Fast Running). Transforming to Matrix-form...")
    else:
        print("Data initialized. Transforming to Matrix-form...")

    input_matrices = trans_matrix_form(train_data)
    print("Transformation completed. Generating sequential graph...")

    laplace_list = graph_construction(input_matrices, item_dict, user_dict)
    print("Graph Initialized. Generating batches...")

    train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size, padding_num=len(item_dict),
                                     is_train=True)
    test_batches = generate_batches(input_data=test_data, batch_size=param.batch_size, padding_num=len(item_dict),
                                    is_train=False)
    print("Batches loaded. Initializing EA_GPT network...")

    num_items = len(item_dict)
    num_users = len(user_dict)

    module = EA_GPT(num_items=num_items, num_users=num_users, laplace_list=laplace_list)
    print("Model Initialized. Start training...")

    with tf.Session(graph=module.graph, config=module.config) as sess:

        module.sess = sess
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        best_score = -1

        for epoch in range(param.num_epochs):
            time_start = time()
            loss_train = train_module(sess=sess, module=module, batches_train=train_batches)
            time_consumption = time() - time_start

            epoch_num = epoch + 1
            print_train(epoch=epoch_num, loss=loss_train, time_consumption=time_consumption)

            if epoch_num % param.eval_verbose == 0:
                test_start = time()
                [rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20] = \
                    evaluate_module(sess=sess, module=module, batches_test=test_batches, eval_length=len(test_data))
                test_consumption = time() - test_start

                print_evaluation(epoch_num, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20, test_consumption)

                if rc_5 >= best_score:
                    best_score = rc_5
                    saver.save(sess, param.check_points, global_step=epoch_num, write_meta_graph=False)
                    print("Recommender performs better, saving current model....")
                    logging.info("Recommender performs better, saving current model....")

            train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size,
                                             padding_num=len(item_dict), is_train=True)

        print("Recommender training finished.")
        logging.info("Recommender training finished.")

        print("All process finished.")
