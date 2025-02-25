def get_data(data_path):
    # out_file = open("train_data_filter.txt", "w")
    with open(data_path, 'r') as file_object:
        lines = file_object.readlines()
        len_min = len(lines[0].strip().split('\t'))
        for line in lines:
            temp_line = ""
            line = line.strip().split('\t')
            # for item in line:
            #     temp = str(item[1:])
            #     temp_line+=temp
            #     temp_line+="\t"
            if len(line) < len_min:
                len_min = len(line)
        print(len_min)
    return


get_data("movie_len_10M.txt")
