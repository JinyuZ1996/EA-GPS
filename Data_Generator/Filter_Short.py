def get_data(data_path):
    out_file = open("../Data_Generator/Video_F.txt", "w")
    with open(data_path, 'r') as file_object:
        mixed_data = []
        lines = file_object.readlines()
        for line in lines:
            temp_sequence = line
            line = line.strip().split('\t')
            if len(line) < 4:
                continue
            else:
                out_file.writelines(temp_sequence)
                # out_file.write("\t")
            # sequence_all = []
            # user = line[0]  # 每行第一个是uid
            # sequence_all.append(dict_U[user])  # 现在混合seq第一个位置上拼上uid
            # for item in line[1:]:  # 从line中的第二项开始，遍历line中的item，从'E241'到'V326'；for循环将line转换为对应的索引列表；
            #     item_info = item.split('|')
            #     item_id = item_info[0]
            #     if item_id in dict_A:
            #         sequence_all.append(dict_A[item_id])
            #     else:
            #         sequence_all.append(dict_B[item_id] + len(dict_A))  # 为了区分序列中的E与V物品，len(itemE)=8367；
            # temp_sequence.append(sequence_all)  # [0]
            # mixed_data.append(temp_sequence)
    return


get_data("../Data_Generator/Video.txt")
