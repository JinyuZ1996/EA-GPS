# 32,633, 76,146 BOOK
# 34,909, 81,455 Movie
# 12,618, 29,444 Food

import numpy as np

write_train = open("../Data/Novel/train_data.txt", "w")
write_test = open("../Data/Novel/test_data.txt", "w")

with open("../Data/DOUBAN/Novel_FS.txt", "r") as file:
    lines = file.readlines()
    counter = 0
    for line in lines:
        num = np.random.randint(1, 100)
        if num > 50 and counter <= 12617:
            write_test.write(line)
            counter += 1
        else:
            write_train.write(line)
