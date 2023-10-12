import pandas as pd

# df = pd.read_csv("../Data_Generator/Ulist.txt", sep='\t', header=None)
# df.to_csv('../Data_Generator/User_Dict.csv', index=False, sep=',', encoding='utf-8', header=None)

df = pd.read_csv("../Data_Generator/Movie_Dict.csv", sep=',', header=None)
df.to_csv('../Data_Generator/Movie_Dict.txt', index=False, sep='\t', encoding='utf-8', header=None)