from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsRegressor

df = pd.read_csv('/Users/kotaro/PycharmProjects/SMBC_signate/output/col_selected.csv')

test = df[df["health"].isna()]
train = df[~df["health"].isna()]
X = train.drop("health", axis=1)
y = train["health"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# knc = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',
#                             leaf_size=10, n_jobs=-1,
#                             p=3, weights="uniform"
#                             )
# knc.fit(X_train, y_train)
# y_pred = knc.predict(X_test)
# print(f"score is {f1_score(y_test, y_pred, average="macro"):.4f}")

# rfc = RandomForestClassifier(max_depth=None, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, n_estimators=50)
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)
# print(f"score is {f1_score(y_test, y_pred, average="macro"):.4f}")

param_grid = {"weights": ["uniform", "distance"],
              "algorithm": ["auto", "ball_tree"],
              "leaf_size": [8, 9, 10, 11, 12],
              "p": [1, 3],
              }
param_grid = {"weights": ["distance"],
              "algorithm": ["auto"],
              "leaf_size": [1, 2],
              "p": [4, 5],
              }
grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
                           param_grid=param_grid,
                           cv=10,
                           verbose=4,
                           scoring='f1_macro',
                           n_jobs=-1)

# grid_search.fit(X, y)
# print("Best Parameters:", grid_search.best_params_)

"""mybest
# knc = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',
#                             leaf_size=10, n_jobs=-1,
#                             p=3, weights="uniform"
#                             )
"""
knc = KNeighborsClassifier(algorithm='auto',
                           leaf_size=1,
                           n_jobs=-1,
                           p=4,
                           weights="distance"
                           )
knc.fit(X, y)
y_pred = knc.predict(test.drop(["health"], axis=1)).astype(int)

df_submission = pd.read_csv("/Users/kotaro/PycharmProjects/SMBC_signate/output/submission.csv", header=None)
df_submission[1] = y_pred
df_submission.to_csv("/Users/kotaro/PycharmProjects/SMBC_signate/output/sub_knc_colselected.csv", header=None, index=None)