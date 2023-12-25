import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/train.csv").reset_index(drop=True)
# for col in train.columns:
#     print(col, list(train[col].unique()))

temp_ls = train["nta"]
temp = {}
for key in temp_ls:
    if key[:2] not in temp:
        temp[key[:2]] = []
    if key[2:] not in temp[key[:2]]:
        temp[key[:2]].append(key[2:])

print(temp.keys())
