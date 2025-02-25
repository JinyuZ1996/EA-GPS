def split_and_create_dicts(input_file, output_file, user_dict_file, item_dict_file, sequence_length=30):
    # Dictionaries to store unique users and items
    user_dict = {}
    item_dict = {}
    user_count = 0
    item_count = 0

    # Open the output file for writing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                parts = line.strip().split('\t')
                user_id = parts[0]
                sequence = parts[1:]

                # Update user dictionary
                if user_id not in user_dict:
                    user_count += 1
                    user_dict[user_id] = user_count

                # Split the sequence into chunks of the specified length
                for i in range(0, len(sequence), sequence_length):
                    chunk = sequence[i:i+sequence_length]
                    # Update item dictionary for items in this chunk
                    for item in chunk:
                        if item not in item_dict:
                            item_count += 1
                            item_dict[item] = item_count
                    # Write the chunk to the output file
                    outfile.write(str(user_dict[user_id]) + '\t' + '\t'.join(chunk) + '\n')

    # Write user and item dictionaries to files
    with open(user_dict_file, 'w', encoding='utf-8') as userfile, open(item_dict_file, 'w', encoding='utf-8') as itemfile:
        for key, value in user_dict.items():
            userfile.write(str(value) + '\t' + key + '\n')
        for key, value in item_dict.items():
            itemfile.write(str(value) + '\t' + key + '\n')

# Example usage
input_file = 'movie_lens_100k.txt'  # Replace with your actual input file path
output_file = 'ML_origin_100k.txt'  # Replace with your desired output file path
user_dict_file = 'user_dict_100k.txt'  # Replace with your desired user dictionary file path
item_dict_file = 'item_dict_100k.txt'  # Replace with your desired item dictionary file path
split_and_create_dicts(input_file, output_file, user_dict_file, item_dict_file)

# Note: This code assumes that the input file is a text file with each line containing a user ID followed by a tab and the sequence of item IDs.
# Please replace '/path/to/your/input.txt', '/path/to/your/output.txt', '/path/to/your/user_dict.txt', and '/path/to/your/item_dict.txt' with the actual paths to your input and output files.
# This code will read the input file, split sequences longer than 30 items, and write the cleaned data to the output file. It will also create user and item dictionaries.
