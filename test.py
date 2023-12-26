import random

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
from sklearn.model_selection import cross_val_score


def open_train():
    train = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/train.csv")
    train = train.drop(["Unnamed: 0"], axis=1)
    return train


def open_test():
    test = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/input/test.csv").drop(["Unnamed: 0"], axis=1)
    return test


def open_sub():
    sample_submit = pd.read_csv('/Users/kotaro/PycharmProjects/SMBC_signate/output/sample_submission.csv', index_col=0,
                                header=None)
    return sample_submit


def get_train_size():
    return len(open_train())


def concat_train_test(train, test):
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    return df


def split_train_test(df, train_size=get_train_size()):
    train, test = df.iloc[:train_size, :], df.iloc[train_size:, :]
    return train, test


def split_target(data):
    Y = data["health"]
    x = data.drop(["health"], axis=1)
    return Y, x


def check_nans(df):
    print(df.isnull().sum())


def get_ymd(df):
    y = df["created_at"].str[:4]
    m = df["created_at"].str[5:7]
    d = df["created_at"].str[8:10]
    df["created_year"] = y.astype(int)
    df["created_month"] = m.astype(int)
    df["created_date"] = d.astype(int)
    df = df.drop(["created_at"], axis=1)
    return df


def curb_loc(df):
    curb_loc = LabelEncoder()
    df["curb_loc"] = curb_loc.fit_transform(df["curb_loc"])
    return df


def print_unique(df, key):
    print(df[key].unique())
    print(df[key].isna().sum() / len(df[key]))


def tree_dbh(df):
    df["tree_dbh"] //= 10
    return df


def health(df):
    df["health"] = df["health"].replace({0: "Fair",
                                         1: "Good",
                                         2: "Poor"})
    df["health"] = df["health"].replace({"Poor": 0, "Fair": 1, "Good": 2})
    return df


def steward(df):
    # steward_labels = LabelEncoder()
    df.loc[df["steward"].isna(), "steward"] = "non"
    df = df.replace({'steward': {'3or4': 3,
                                 "non": 0,
                                 "1or2": 1,
                                 "4orMore": 4}})
    # df["steward"] = steward_labels.fit_transform(df["steward"])
    return df


def staff(df):
    df["staff"] = df["user_type"].replace({'Volunteer': False,
                                           'NYC Parks Staff': True,
                                           'TreesCount Staff': False
                                           })
    return df


def guards(df):
    df.loc[df["guards"].isna(), "guards"] = "Unsure"
    df["guards"] = df["guards"].replace({'Helpful': 1, 'Harmful': -1, 'Unsure': 0})
    return df


def sidewalk(df):
    sidewalk_labels = LabelEncoder()
    df["sidewalk"] = sidewalk_labels.fit_transform(df["sidewalk"])
    return df


def user_type(df):
    user_type_labels = LabelEncoder()
    df["user_type"] = user_type_labels.fit_transform(df["user_type"])
    return df


temp_lst = []


def problems(df):
    def get_problems(key):
        if type(key) == float:
            return "not"
        else:
            lst = []
            temp = ""
            for i in range(len(key)):
                if key[i] == key[i].upper():
                    if (temp != ""):
                        if temp != "Other":
                            lst.append(temp)
                        temp = ""
                temp += key[i]
            lst.append(temp)
            temp_lst.append(lst)
            return lst

    df.loc[:, "problems"] = df.loc[:, "problems"].apply(get_problems)
    df.loc[:, "problems_num"] = df.loc[:, "problems"].apply(lambda x: len(x))
    return df


def problems_category(df):
    labels = ['Stones', 'Branch', 'Lights', 'Wires', 'Rope', 'Metal', 'Grates', 'Trunk', 'Sneakers']
    for label in labels:
        df["problems_" + label] = 0
        for i in range(len(df)):
            if label in df.loc[i, "problems"]:
                df.loc[i, "problems_" + label] = 1
    df = df.drop("problems", axis=1)
    return df


def spc_common(df):
    spc_labels = LabelEncoder()
    df["spc_common"] = spc_labels.fit_transform(df["spc_common"])
    return df


def spc_latin(df):
    spc_latin_labels = LabelEncoder()
    df["spc_latin"] = spc_latin_labels.fit_transform(df["spc_latin"])
    return df


def season(df):
    if "created_month" not in df.columns:
        return df
    else:
        df["season"] = df["created_month"].replace({1: "Winter",
                                                    2: "Winter",
                                                    3: "Spring",
                                                    4: "Spring",
                                                    5: "Spring",
                                                    6: "Spring",
                                                    7: "Summer",
                                                    8: "Summer",
                                                    9: "Autumn",
                                                    10: "Autumn",
                                                    11: "Autumn",
                                                    12: "Winter"})
        df["season"] = df["season"].replace({"Autumn": 0,
                                             "Winter": 1,
                                             "Spring": 2,
                                             "Summer": 3,
                                             })
        df = df.drop("created_month", axis=1)
        return df


def nta(df):
    df["nta"] = df["nta"].str[:2]
    nta_labels = LabelEncoder()
    df["nta"] = nta_labels.fit_transform(df["nta"])
    return df


def nta_name(df):
    # nta_name_labels = LabelEncoder()
    # df["nta_name"] = nta_name_labels.fit_transform(df["nta_name"])
    df = df.drop("nta_name", axis=1)
    return df


def boroname(df):
    boroname_labels = LabelEncoder()
    df["boroname"] = boroname_labels.fit_transform(df["boroname"])
    return df


def boro_ct(df):
    df["boro_ct"] //= 1000000
    return df


def zip_city(df):
    zip_city_labels = LabelEncoder()
    df["zip_city"] = zip_city_labels.fit_transform(df["zip_city"])
    return df


def select_columns(df,
                   by_num=False,
                   num=10,
                   by_cols=False,
                   cols={"health": True, "tree_dbh": False}
                   ):
    if by_cols:
        use_cols = []
        for key in cols:
            if cols[key]:
                use_cols.append(key)
        df = df[use_cols]
        for col in df.columns:
            if col not in use_cols:
                print(f"{col} is dropped!")
    elif by_num:
        for col in df.columns:
            if len(df[col].unique()) >= num:
                df = df.drop(col, axis=1)
                print(f"{col} is dropped!")
    return df


from tools.select_columns import select_columns_random

dics = []
def processing(not_process=False):
    train = open_train()
    test = open_test()
    df = concat_train_test(train, test)
    if not_process:
        return df
    else:
        df = health(df)
        df = get_ymd(df)
        df = tree_dbh(df)
        df = season(df)
        df = curb_loc(df)
        df = steward(df)
        df = guards(df)
        df = sidewalk(df)
        df = user_type(df)
        df = staff(df)
        df = problems(df)
        df = problems_category(df)
        df = spc_common(df)
        df = spc_latin(df)
        df = nta(df)
        df = nta_name(df)
        df = boroname(df)
        df = boro_ct(df)
        df = zip_city(df)
        cols = {'boro_ct': False,
                'borocode': False,
                'boroname': True,
                'cb_num': False,
                'cncldist': False,
                'created_date': False,
                'created_year': True,
                'curb_loc': False,
                'guards': True,
                'health': True,
                'nta': False,
                'problems_Branch': True,
                'problems_BranchOther': True,
                'problems_Grates': True,
                'problems_Lights': True,
                'problems_Metal': True,
                'problems_RootOther': True,
                'problems_Rope': False,
                'problems_Sneakers': True,
                'problems_Stones': False,
                'problems_Trunk': True,
                'problems_TrunkOther': False,
                'problems_Wires': True,
                'problems_num': True,
                'season': False,
                'sidewalk': True,
                'spc_common': True,
                'spc_latin': False,
                'st_assem': True,
                'st_senate': True,
                'staff': True,
                'steward': True,
                'tree_dbh': False,
                'user_type': False,
                'zip_city': False}
        cols = {'boro_ct': True,
 'borocode': True,
 'boroname': True,
 'cb_num': False,
 'cncldist': True,
 'created_date': False,
 'created_year': True,
 'curb_loc': False,
 'guards': False,
 'health': True,
 'nta': False,
 'problems_Branch': True,
 'problems_Grates': False,
 'problems_Lights': True,
 'problems_Metal': True,
 'problems_Rope': True,
 'problems_Sneakers': True,
 'problems_Stones': False,
 'problems_Trunk': False,
 'problems_Wires': True,
 'problems_num': False,
 'season': True,
 'sidewalk': True,
 'spc_common': True,
 'spc_latin': False,
 'st_assem': True,
 'st_senate': True,
 'staff': False,
 'steward': True,
 'tree_dbh': False,
 'user_type': False,
 'zip_city': True}
        cols = {'tree_dbh': True, 'curb_loc': False, 'health': True, 'steward': True, 'guards': True, 'sidewalk': True, 'user_type': False, 'spc_common': False, 'spc_latin': False, 'nta': True, 'borocode': False, 'boro_ct': True, 'boroname': False, 'zip_city': True, 'cb_num': True, 'st_senate': True, 'st_assem': False, 'cncldist': False, 'created_year': True, 'created_date': False, 'season': False, 'staff': True, 'problems_num': True, 'problems_Stones': True, 'problems_Branch': False, 'problems_Lights': False, 'problems_Wires': False, 'problems_Rope': True, 'problems_Metal': True, 'problems_Grates': False, 'problems_Trunk': False, 'problems_Sneakers': True}

        # cols = {'tree_dbh': True,
        #         'curb_loc': True,
        #         'health': True,
        #         'steward': True,
        #         'guards': True,
        #         'sidewalk': True,
        #         'user_type': True,
        #         'spc_common': False,
        #         'spc_latin': False,
        #         'nta': True,
        #         'borocode': True,
        #         'boro_ct': False,
        #         'boroname': False,
        #         'zip_city': False,
        #         'cb_num': False,
        #         'st_senate': False,
        #         'st_assem': False,
        #         'cncldist': False,
        #         'created_year': True,
        #         'created_date': False,
        #         'season': True,
        #         'staff': True,
        #         'problems_Stones': True,
        #         'problems_Branch': True,
        #         'problems_Lights': True,
        #         'problems_TrunkOther': True,
        #         'problems_Wires': True,
        #         'problems_Rope': True,
        #         'problems_Metal': True,
        #         'problems_Grates': True,
        #         'problems_RootOther': True,
        #         'problems_BranchOther': True,
        #         'problems_Trunk': True,
        #         'problems_Sneakers': True,
        #         "problems_num": True,
        #         }
        df = select_columns(df, by_cols=True, cols=cols)
        # df, dic = select_columns_random(df)
        # dics.append(dic)
        return df


def RFC(df_concat):
    train, test = split_train_test(df_concat, train_size=get_train_size())
    Y, x = split_target(train)
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred2 = pred.tolist()
    print(f"Poor:{pred2.count(0)}")
    print(f"Fair:{pred2.count(1)}")
    print(f"Good:{pred2.count(2)}")
    # score = accuracy_score(y_test, pred)
    score = f1_score(y_test, pred, average='macro')
    print(f"F1 Score：{score * 100:.3f}%")
    return score


def output(df_concat):
    train, test = split_train_test(df_concat, train_size=get_train_size())

    Y, x = split_target(train)
    Y_test, x_test = split_target(test)

    model = RandomForestClassifier(random_state=42)
    model.fit(x, Y)
    pred = model.predict(x_test)
    pred2 = []
    for i in range(len(pred)):
        if pred[i] == 0:
            pred2.append(2)
        elif pred[i] == 1:
            pred2.append(0)
        else:
            pred2.append(1)
    sample_submit = open_sub()
    sample_submit[1] = pred2
    print(f"Poor:{pred2.count(2)}")
    print(f"Fair:{pred2.count(0)}")
    print(f"Good:{pred2.count(1)}")
    sample_submit.to_csv('submission.csv', header=None)

def closs_RFC(df,verbose=False,cv=5):
    model = RandomForestClassifier(random_state=42)
    train, test = split_train_test(df, train_size=get_train_size())
    Y, x = split_target(train)
    scores = cross_val_score(model, np.array(x), np.array(Y), scoring="f1_macro", cv=cv)
    if verbose:
        print(f'Cross-Validation scores: {scores}')
        print(f'Average score: {np.mean(scores):.3f}')
    return np.mean(scores)

def change_col_col(temp_cols,other_cols,target="health",n=10):
    if (np.random.rand() <= 1/(2*n+1)):
        cols = temp_cols + other_cols
        temp_cols = random.sample(cols, random.randint(1, len(cols)))
        other_cols = []
        if target not in temp_cols:
            temp_cols.append(target)
        for key in cols:
            if key not in temp_cols:
                other_cols.append(key)
        return temp_cols, other_cols
    elif (1/(2*n+1) < np.random.rand() <= (n+1)/(2*n+1)) or (len(other_cols) == 0):
        c = random.choice(temp_cols)
        if c == target:
            return temp_cols, other_cols
        temp_cols.remove(c)
        other_cols.append(c)
    else:
        c = random.choice(other_cols)
        other_cols.remove(c)
        temp_cols.append(c)
    return temp_cols, other_cols

def select_columns_RFC(df, cnt=20):
    cols = list(df.columns)
    target = "health"
    temp_cols = random.sample(cols,random.randint(1, len(cols)))
    other_cols = []
    if target not in temp_cols:
        temp_cols.append(target)
    for key in cols:
        if key not in temp_cols:
            other_cols.append(key)
    df_temp = df[temp_cols]
    max_score = closs_RFC(df_temp)
    max_cols = temp_cols
    temp = 1
    while temp <= cnt:
        temp_cols_next, other_cols_next = change_col_col(temp_cols, other_cols)
        df_temp = df[temp_cols_next]
        current_score = closs_RFC(df_temp)
        if max_score < current_score:
            max_score = current_score
            max_cols = temp_cols
            temp_cols = temp_cols_next
            other_cols = other_cols_next
            print(temp_cols)
        print(f"{temp}/{cnt} done! \n Current score: {current_score*100:.3f} \n Max score: {max_score*100:.3f}")
        temp += 1
    print(f"{max_score:.4f}")
    print(max_cols)
    return df[max_cols]








if __name__ == "__main__":
    # scores = []
    # for i in range(10):
    #     print(f"試行: {i+1}回目")
    #     df_concat = processing()
    #     score = RFC(df_concat)
    #     scores.append(score)
    # print(scores)

    # scores = []
    # for i in range(10):
    #     df_concat = processing()
    #     score = closs_RFC(df_concat)
    #     scores.append(score)
    #     print(f"試行: {i+1}回目 {score:.3f}")
    # print(scores)
    # d = np.argmax(scores)
    # print(f"MaxScore: {scores[d]}")
    # print(dics[d])
    # print(dics)

    # df_concat = processing()
    # closs_RFC(df_concat)

    df_concat = processing()
    RFC(df_concat)
    output(df_concat)
