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


def steward(df):
    steward_labels = LabelEncoder()
    df.loc[df["steward"].isna(), "steward"] = "non"
    df["steward"] = steward_labels.fit_transform(df["steward"])
    return df


def guards(df):
    guard_labels = LabelEncoder()
    df.loc[df["guards"].isna(), "guards"] = "non"
    df["guards"] = guard_labels.fit_transform(df["guards"])
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
                    if (temp != "") and (key[i] != "O"):
                        lst.append(temp)
                        temp = ""
                temp += key[i]
            lst.append(temp)
            temp_lst.append(lst)
            return lst

    df.loc[:, "problems"] = df.loc[:, "problems"].apply(get_problems)
    return df


def problems_category(df):
    labels = ['Stones', 'Branch', 'Lights', 'TrunkOther', 'Wires', 'Rope', 'Metal', 'Grates', 'RootOther',
              'BranchOther', 'Trunk', 'Sneakers']
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

def select_columns(df,num):
    for col in df.columns:
        if len(df[col].unique()) >= num:
            df = df.drop(col, axis=1)
            print(f"{col} is dropped!")
    return df

def processing():
    train = open_train()
    test = open_test()
    df = concat_train_test(train, test)
    df = get_ymd(df)
    df = curb_loc(df)
    df = steward(df)
    df = guards(df)
    df = sidewalk(df)
    df = user_type(df)
    df = problems(df)
    df = problems_category(df)
    df = spc_common(df)
    df = spc_latin(df)
    df = nta(df)
    df = nta_name(df)
    df = boroname(df)
    df = boro_ct(df)
    df = zip_city(df)
    df = select_columns(df, 13)
    return df


def RFC(df_concat):
    train, test = split_train_test(df_concat, train_size=get_train_size())
    Y, x = split_target(train)
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred2 = pred.tolist()
    print(f"0:{pred2.count(0)}")
    print(f"1:{pred2.count(1)}")
    print(f"2:{pred2.count(2)}")
    # score = accuracy_score(y_test, pred)
    score = f1_score(y_test, pred, average='macro')
    print(f"正解率：{score * 100}%")


def output(df_concat):
    train, test = split_train_test(df_concat, train_size=get_train_size())

    Y, x = split_target(train)
    Y_test, x_test = split_target(test)

    model = RandomForestClassifier(random_state=42)
    model.fit(x, Y)
    pred = model.predict(x_test)

    sample_submit = open_sub()
    sample_submit[1] = pred
    pred2 = pred.tolist()
    print(f"sub0:{pred2.count(0)}")
    print(f"sub1:{pred2.count(1)}")
    print(f"sub2:{pred2.count(2)}")
    sample_submit.to_csv('submission.csv', header=None)


if __name__ == "__main__":
    df_concat = processing()
    RFC(df_concat)
    # output(df_concat)