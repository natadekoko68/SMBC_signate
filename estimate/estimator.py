import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from preprocessing.arrangement import split_train_test,get_train_size,split_target,open_sub,processing


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

def change_col_col(temp_cols,other_cols, current_score, target="health",n=10,objective_score=32):
    if objective_score-1 >= current_score:
        cols = temp_cols + other_cols
        temp_cols = random.sample(cols, random.randint(1, len(cols)))
        other_cols = []
        if target not in temp_cols:
            temp_cols.append(target)
        for key in cols:
            if key not in temp_cols:
                other_cols.append(key)
        return temp_cols, other_cols
    elif np.random.choice([True, False]) or (len(other_cols) == 0):
        c = random.choice(temp_cols)
        if c == target:
            pass
        elif len(temp_cols) > 1:
            temp_cols.remove(c)
            other_cols.append(c)
        else:
            pass

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
    current_score = max_score
    max_cols = temp_cols
    temp = 1
    while temp <= cnt:
        temp_cols_next, other_cols_next = change_col_col(temp_cols, other_cols, current_score)
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
    print(f"Total maxscore: {max_score*100:.4f}")
    print(max_cols)
    return df[max_cols]