import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    df.loc[df["steward"].isna(), "steward"] = "non"
    df = df.replace({'steward': {'3or4': 3,
                                 "non": 0,
                                 "1or2": 1,
                                 "4orMore": 4}})
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
        cols = {'created_year': True, 'curb_loc': False, 'tree_dbh': True, 'health': True, 'cb_num': True, 'user_type': False, 'season': False, 'problems_Branch': False, 'problems_Sneakers': True, 'problems_Rope': False, 'st_senate': False, 'cncldist': False, 'problems_Trunk': False, 'spc_common': False, 'problems_num': True, 'problems_Stones': True, 'staff': True, 'sidewalk': True, 'boro_ct': True, 'problems_Metal': True, 'guards': True, 'steward': True}

        df = select_columns(df, by_cols=True, cols=cols)
        return df