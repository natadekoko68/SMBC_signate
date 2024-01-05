import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pprint
import seaborn as sns

train = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/train.csv")

rare_num = 10

classified = {}

for spc_common in train["spc_common"].unique():
    temp_df = train[train["spc_common"] == spc_common]
    if len(temp_df) > rare_num:
        temp = {}
        for num in temp_df["health"].unique():
            temp[num] = temp_df["health"].value_counts()[num]
        for num in [0, 1, 2]:
            if num not in temp:
                temp[num] = 0
        class_temp = round((temp[2])/(temp[0]+temp[1]+temp[2]), 2)
        classified[spc_common] = class_temp
    else:
        classified[spc_common] = "rare"

print(classified)




