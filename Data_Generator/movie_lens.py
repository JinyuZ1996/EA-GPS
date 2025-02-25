import pandas as pd
import csv

# 读取制表符分隔的数据文件
# input_file = "ratings.txt"  # 替换为你的输入文件名
# df = pd.read_csv(input_file, sep="::", header=None, names=["user_id", "product", "rating", "timestamp"])
#
# # 数据预处理：排序
# df_sorted = df[df["rating"] == 5].sort_values(by=["user_id", "timestamp"])
#
# # 构造交互序列
# df_grouped = df_sorted.groupby("user_id")["product"].apply(list).reset_index(name="interaction_sequence")
#
# # 确保交互序列是字符串类型
# df_grouped["interaction_sequence"] = df_grouped["interaction_sequence"].apply(lambda x: "\t".join(map(str, x)))
#
# # 过滤序列：移除序列中元素少于3个的行
# df_grouped = df_grouped[df_grouped["interaction_sequence"].str.len() >= 3]
#
# # 输出处理后的数据到新的文件，不包含表头，不添加引号，设置转义字符
# output_file_filtered = "m_lens10M.txt"  # 替换为你的输出文件名
# df_grouped.to_csv(output_file_filtered, index=False, header=False, sep="\t")

########################
# import csv


# def remove_quotes_from_sequence_tab_delimited(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8',
#                                                                  newline='') as outfile:
#         for line in infile:
#             # Split the line by tab
#             parts = line.strip().split('\t')
#             # Assuming the first part is the user ID
#             user_id = parts[0]
#             # Process the sequence, removing quotes
#             sequence = [item.replace('"', '') for item in parts[1:]]
#             # Write the processed row to the output file
#             outfile.write(user_id + '\t' + '\t'.join(sequence) + '\n')
#
#
# # Example usage
# input_file = 'processed_data.txt'  # Replace with your actual input file path
# output_file = 'movie_lens_100k.txt'  # Replace with your desired output file path
# remove_quotes_from_sequence_tab_delimited(input_file, output_file)

# import pandas as pd

# # 读取数据
# input_file = "ratings.txt"  # 替换为您的输入文件名
# df = pd.read_csv(input_file, sep="::", header=None, names=["user_id", "item_id", "rating", "timestamp"])
#
# # 转换数据类型
# df["user_id"] = df["user_id"].astype(int)
# df["item_id"] = df["item_id"].astype(int)
# df["rating"] = df["rating"].astype(int)
# df["timestamp"] = df["timestamp"].astype(int)
#
# # 保留评分等于5的交互
# df_filtered = df[df["rating"] == 5]
#
# # 按照时间顺序对每个用户的交互进行排序
# df_sorted = df_filtered.sort_values(by=["user_id", "timestamp"])
#
# # 将每个用户的交互序列整理成一行
# df_grouped = df_sorted.groupby("user_id")["item_id"].apply(list).reset_index(name="interaction_sequence")
#
# # 确保每个用户的交互序列长度不小于3
# df_grouped = df_grouped[df_grouped["interaction_sequence"].apply(len) >= 3]
#
# # 如果某个用户的交互序列超过30个，将其分为多个序列
# def split_sequence(sequence, max_length=30):
#     return [sequence[i:i + max_length] for i in range(0, len(sequence), max_length)]
#
# df_grouped["interaction_sequence"] = df_grouped["interaction_sequence"].apply(split_sequence)
#
# # 展平序列
# df_flattened = pd.DataFrame([(user_id, seq) for user_id, sequences in df_grouped.values for seq in sequences],
#                             columns=["user_id", "interaction_sequence"])
#
# # 将序列转换为字符串，每个元素之间用制表符分隔
# df_flattened["interaction_sequence"] = df_flattened["interaction_sequence"].apply(lambda x: "\t".join(map(str, x)))
#
# # 输出处理后的数据到新的文件
# output_file = "ml_10M_origin.txt"  # 替换为您的输出文件名
# df_flattened.to_csv(output_file, index=False, header=False, sep="\t")

# Implementing the solution without using Pandas DataFrame

# Modifying the function to handle non-integer ratings gracefully

def process_data(input_file, output_file):
    # Dictionary to store interaction sequences for each user
    user_sequences = {}

    # Read and process the input file
    with open(input_file, 'r') as file:
        for line in file:
            try:
                user_id, item_id, rating, timestamp = line.strip().split("::")
                user_id, rating, timestamp = int(user_id), float(rating), int(timestamp)

                # Filter interactions with rating 5
                if rating == 5:
                    if user_id not in user_sequences:
                        user_sequences[user_id] = []
                    user_sequences[user_id].append((item_id, timestamp))
            except ValueError:
                # Skip lines with non-integer ratings or other format issues
                continue

    # Sort and split sequences for each user
    for user_id in user_sequences:
        # Sort by timestamp
        user_sequences[user_id].sort(key=lambda x: x[1])
        # Extract item_ids after sorting
        user_sequences[user_id] = [item_id for item_id, timestamp in user_sequences[user_id]]

        # Split sequences if they are longer than 30
        user_sequences[user_id] = [user_sequences[user_id][i:i + 30] for i in range(0, len(user_sequences[user_id]), 30)]

    # Write to the output file
    with open(output_file, 'w') as file:
        for user_id, sequences in user_sequences.items():
            for sequence in sequences:
                # Ensure each sequence has at least 3 elements
                if len(sequence) >= 5:
                    file.write(f"{user_id}\t" + "\t".join(sequence) + "\n")

# Sample input and output file names (replace with your actual file names)
input_file = "ratings.txt"
output_file = "movie_len_10M.txt"

# Call the function to process the data
process_data(input_file, output_file)

# Return the content of the output file for demonstration
with open(output_file, 'r') as file:
    output_content = file.read()





