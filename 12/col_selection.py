import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/Users/kotaro/PycharmProjects/SMBC_signate/input/using_data.csv')

train = df[~df["health"].isna()]
test = df[df["health"].isna()]

X = train.drop(["health"], axis=1)
y = train["health"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

feature_importances = model.feature_importances_

importances_dict = {}

for i in range(len(X.columns)):
    # print(f"{X.columns[i]}: {feature_importances[i]}")
    importances_dict[X.columns[i]] = feature_importances[i]

score_sorted = sorted(importances_dict.items(), key=lambda x: x[1])

# print(score_sorted)

for i in range(len(score_sorted)-1, -1, -1):
    print(f"{score_sorted[i][0]}: {score_sorted[i][1]}")

# print(len(score_sorted))
use_cols = []
for i in range(len(score_sorted)):
    use_cols.append(score_sorted[i][0])
use_cols.append("health")
use_data = df[use_cols]
print(use_data.head())
use_data.to_csv("/Users/kotaro/PycharmProjects/SMBC_signate/output/col_selected.csv", index=False)

