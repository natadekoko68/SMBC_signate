import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pprint
import seaborn as sns

train = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/train.csv")
# print(train)

# test = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/test.csv")
# print(test)


col = 3
row = 3

fig = plt.figure()
cnt = 1
for key in train["spc_common"].unique():
    temp = train[train["spc_common"] == key]
    if len(temp) >= 2:
        ax = fig.add_subplot(col, row, cnt)
        labels_num = temp["health"].unique()
        cnts = []
        labels_char = []
        colors = []
        for label in labels_num:
            num = temp["health"].value_counts(label)
            cnts.append(num[label])
            if label == 0:
                labels_char.append("Fair")
                colors.append("green")
            elif label == 1:
                labels_char.append("Good")
                colors.append("yellow")
            elif label == 2:
                labels_char.append("Poor")
                colors.append("red")
        plt.pie(cnts, labels=labels_char, colors=colors ,counterclock=False, startangle=90)
        plt.title(key)
        plt.tight_layout()
        cnt += 1
        if cnt > col*row:
            plt.show()
            plt.close()
            fig = plt.figure()
            cnt -= col*row
plt.show()


# sns.countplot(data=train, x="spc_common")
# plt.show()