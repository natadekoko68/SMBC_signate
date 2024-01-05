import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pprint
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import sys

train = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/train.csv").drop(["Unnamed: 0"], axis=1)
test = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/test.csv").drop(["Unnamed: 0"], axis=1)
sample_submit = pd.read_csv('/Users/kotaro/PycharmProjects/SMBC_signate/output/sample_submission.csv', index_col=0,
                            header=None)

df = pd.concat([train, test], axis=0).reset_index(drop=True)
# df = train

# arrange created_at
df["year"] = df["created_at"].str[0:4].astype(int)
df["month"] = df["created_at"].str[5:7].astype(int)
df["day"] = df["created_at"].str[8:].astype(int)
df = df.drop("created_at", axis=1)


# arrange curb_loc
def curb_loc_encoder(x):
    if x == "OnCurb":
        return 0
    elif x == "OffsetFromCurb":
        return 1
    else:
        return np.nan


df["curb_loc"] = df["curb_loc"].apply(curb_loc_encoder)


# arrange steward
def steward_encoder(x):
    if x == "1or2":
        return 1
    elif x == "3or4":
        return 2
    elif x == "4orMore":
        return 3
    else:
        return 0


def myisna(x):
    if type(x) == str:
        return False
    else:
        return True


df["steward_nan"] = df["steward"].apply(myisna)
df["steward"] = df["steward"].apply(steward_encoder)


# arrange guards

def guard_encoder(x):
    if x == "Helpful":
        return 1
    elif x == "Harmful":
        return 2
    elif x == "Unsure":
        return 3
    else:
        return 0


df["guards_nan"] = df["guards"].apply(myisna)
df["guards"] = df["guards"].apply(guard_encoder)


# arrange sidewalk
def sidewalk_encoder(x):
    if x == "Damage":
        return 1
    elif x == "NonDamage":
        return 2
    else:
        return 0


df["sidewalk"] = df["sidewalk"].apply(sidewalk_encoder)


# arrange user_type
# print(df["user_type"])
def user_type_encoder(x):
    if x == "Volunteer":
        return 1
    elif x == "NYC Parks Staff":
        return 2
    elif x == "TreesCount Staff":
        return 3
    else:
        return 0


df["user_type"] = df["user_type"].apply(user_type_encoder)

# arrange spc_common
spc_common_counts = df["spc_common"].value_counts()
for i in range(len(df)):
    df.loc[i, "spc_common_num"] = spc_common_counts[df.loc[i, "spc_common"]]

# arrange spc_latin
df = df.drop(["spc_latin"], axis=1)

# arrange problems
problems_lst = []


def splitter(x):
    if type(x) == float:
        return ["unknown"]
    else:
        lst = []
        temp = ""
        for i in range(len(x)):
            if x[i] == x[i].upper():
                if (temp != ""):
                    if temp != "Other":
                        lst.append(temp)
                        if temp not in problems_lst:
                            problems_lst.append(temp)
                    temp = ""
            temp += x[i]
        if temp != "Other":
            lst.append(temp)
            if temp not in problems_lst:
                problems_lst.append(temp)
        return lst


df["problems"] = df["problems"].apply(splitter)

df["problems_num"] = df["problems"].apply(len)

for key in problems_lst:
    for i in range(len(df)):
        if key in df.loc[i, "problems"]:
            df.loc[i, "problems" + f"_{key}"] = 1
        else:
            df.loc[i, "problems" + f"_{key}"] = 0

df = df.drop(["problems"], axis=1)

# arrange nta
df["nta_first_letter"] = df["nta"].str[0]
df["nta_alphabets"] = df["nta"].str[:2]
df["nta_first_number"] = df["nta"].str[2].astype(int)
df["nta_numbers"] = df["nta"].str[2:].astype(int)
df = df.drop(["nta"], axis=1)

# arrange nta_name
df = df.drop(["nta_name"], axis=1)

# arrange boroname
df = df.drop(["boroname"], axis=1)

# arrange borocode
df = df.drop(["borocode"], axis=1)

# arrange cb_num
df = df.drop(["cb_num"], axis=1)

# arrange st_assem
df = df.drop(["st_assem"], axis=1)

df = df.drop(["nta_alphabets", "zip_city"], axis=1)

df = pd.get_dummies(df)

df.to_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/using_data.csv", index=False)


# prediction
test = df.loc[len(train):, :]
df = df.loc[:len(train)-1, :]
X = df.drop(["health"], axis=1)
y = df["health"]



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# tuned_parameters = [{"n_neighbors": [1, 5, 9],
#                      "weights": ["uniform", "distance"],
#                      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
#                      "leaf_size": [10, 30, 50, 100],
#                      "p": [3, 1, 2],
#                      "n_jobs": [-1],
#                      }]

# clf = GridSearchCV(KNeighborsClassifier(),
#                    tuned_parameters,
#                    cv=10,
#                    verbose=4,
#                    scoring="f1_macro")
#
# clf.fit(X_train, y_train)
# print(clf.best_params_)

"""mybestscore
clf6 = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',
                            leaf_size=10, n_jobs=-1,
                            p=3, weights="uniform"
                            )

clf6.fit(X, y)
y_pred6 = clf6.predict(test.drop(["health"], axis=1)).astype(int)

df_submission = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/output/submission.csv", header=None)
df_submission[1] = y_pred6
df_submission.to_csv("/Users/kotaro/PycharmProjects/SMBC_signate/output/sub_knc.csv", header=None, index=None)
"""
