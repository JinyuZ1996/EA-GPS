# def get_data(data_path):
#     out_file = open("../Data_Generator/Novel_FS.txt", "w")
#     with open(data_path, 'r') as file_object:
#         lines = file_object.readlines()
#         for line in lines:
#             temp_line = ""
#             line = line.strip().split('\t')
#             for item in line:
#                 temp = str(item[1:])
#                 temp_line+=temp
#                 temp_line+="\t"
#             temp_line+="\n"
#             out_file.write(temp_line)
#     return

def get_data(data_path):
    out_file = open("../Data_Generator/User_dict_FS.txt", "w")
    with open(data_path, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            temp_line = ""
            line = line.strip().split('\t')
            # for item in line:
            #     temp = str(item[1:])
            #     temp_line+=temp
            #     temp_line+="\t"
            temp_line+=str(line[0])
            temp_line += "\t"
            temp_line += str(line[1][1:])
            temp_line+="\n"
            out_file.write(temp_line)
    return


get_data("../Data_Generator/U_dict.txt")
