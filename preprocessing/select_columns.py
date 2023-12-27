import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pprint

def select_columns_random(df):
    dic = {}
    use_cols = []
    for i in df.columns:
        if i == "health":
            dic[i] = True
            use_cols.append(i)
        else:
            a = random.choice([True, False])
            dic[i] = a
            if a:
                use_cols.append(i)
    # pprint.pprint(dic)
    return df[use_cols], dic




