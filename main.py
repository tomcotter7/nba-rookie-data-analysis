# %% import statements
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# %% division function
def weird_division(n, d):
    return n / d if d else 0.0


# %% Read data in
df = pd.read_csv("data/nba_logreg.csv")
df.rename(columns={'3P Made': '3PM'}, inplace=True)
df[['TARGET_5Yrs']] = df[['TARGET_5Yrs']].astype(int).astype(str)

# %% EDA
# print(df.head())
# print(df.info())  # - from this I noticed that 3P% is missing 11 values
print(df.isnull().sum())
print(df.isna().sum())

# %% Data Imputation
# Let's impute the missing values. We know that 3P% is 3PM / 3PA * 100
df['3P%'] = df.apply(lambda row: (
    weird_division(row['3PM'], row['3PA'])) * 100, axis=1)

# Creating some interesting graphs

# %% Create a correlation matrix for all the variables
corr = df[['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%',
           'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
           'TOV']].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
heatmap.set(title="Correlation of features")
fig = heatmap.get_figure()
fig.savefig("graphs/heatmap.png")


# %% Compare some of the highly correlated variables split by the target variable
fig, axes = plt.subplots(1, 3, figsize=(15, 15))
fig.suptitle("Comparision of metrics split by 5 Years")

sns.scatterplot(ax=axes[0], x="PTS", y="MIN", hue="TARGET_5Yrs", data=df)
axes[0].set_title("Points vs Minutes")

sns.scatterplot(ax=axes[1], x="AST", y="TOV", hue="TARGET_5Yrs", data=df)
axes[1].set_title("Assist vs TurnOvers")

sns.scatterplot(ax=axes[2], x="REB", y="BLK", hue="TARGET_5Yrs", data=df)
axes[2].set_title("Rebounds vs Blocks")

plt.show()
fig.savefig("graphs/comparisions.png")

# Now let's split the data into test and training and build some models.
# The models I plan to use are:
# SVM
# Random Forest

# %% Split the data into features and labels
# Also, remove the name of the player, as this does not matter
x, y = df.iloc[:, 1:-1], df.iloc[:, [-1]].astype(int)
# Normalize the data
scaler = StandardScaler()
x = scaler.fit_transform(x.to_numpy())
print(x)
y = y.to_numpy()


# %% BayesianSearch to optimize hyper-params
space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400, 500, 600]),
    "max_depth": hp.quniform("max_depth", 1, 15, 1),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}


# define objective function
def hyperparameter_tuning(space):
    clf = RandomForestClassifier(
        n_estimators=space['n_estimators'], criterion=space['criterion'],
        max_depth=int(space['max_depth']), n_jobs=-1)
    acc = cross_val_score(clf, x, y, scoring="accuracy").mean()
    return {"loss": -acc, "status": STATUS_OK}


trials = Trials()
best = fmin(
    fn=hyperparameter_tuning,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)
print("Best: {}".format(best))

# The results can be seen if you run it, but from my results
# Max_depth - 6.0, criterion - entropy, n_estimators - 200, with a 69.7% accuracy.

# %% Do the same with svm (BayesianSearch to optimize hps)

space = {
      'C': hp.choice('C', np.arange(0.005, 1.0, 0.01)),
      'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf']),
      'degree': hp.choice('degree', [2, 3, 4]),
      'probability': hp.choice('probability', [True])
      }


def hyperparameter_tuning_svm(space):
    clf = svm.SVC(**space)
    acc = cross_val_score(clf, x, y, scoring="accuracy").mean()
    return {"loss": -acc, "status": STATUS_OK}


trials = Trials()
best = fmin(
    fn=hyperparameter_tuning_svm,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)
print("Best: {}".format(best))

# Again, results can be seen if ran, but from my results:
# C - 0.645, degree - 4, kernel - linear, probability - True

# %% Finally, let's do this with an XGBoost Classifier


xgboost_space = {
            'max_depth': hp.choice('x_max_depth', [2, 3, 4, 5, 6]),
            'min_child_weight': hp.choice('x_min_child_weight', np.round(np.arange(0.0, 0.2, 0.01), 5)),
            'learning_rate': hp.choice('x_learning_rate', np.round(np.arange(0.005, 0.3, 0.01), 5)),
            'subsample': hp.choice('x_subsample', np.round(np.arange(0.1, 1.0, 0.05), 5)),
            'colsample_bylevel': hp.choice('x_colsample_bylevel', np.round(np.arange(0.1, 1.0, 0.05), 5)),
            'colsample_bytree': hp.choice('x_colsample_bytree', np.round(np.arange(0.1, 1.0, 0.05), 5)),
            'n_estimators': hp.choice('x_n_estimators', np.arange(25, 100, 5))
}


def hyperparameter_tuning_xgb(space):
    clf = XGBClassifier(**space, n_jobs=-1)
    acc = cross_val_score(clf, x, y, scoring="accuracy").mean()
    return {"loss": -acc, "status": STATUS_OK}


trials = Trials()
best = fmin(
    fn=hyperparameter_tuning_xgb,
    space=xgboost_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

print("Best: {}", format(best))

# My results show:
# max_depth - 3, min_child_weight - 0.01, learning_rate - 0.045, subsample - 0.75,
# colsample_bylevel - 0.5 ,colsample_bytree - 0.75, n_estimators - 95


# %% Let's train the model fully ?

# We should use XGBoost, that seemed to give the best results

xgboost = XGBClassifier(max_depth=3, min_child_weight=0.01, learning_rate=0.045,
                        subsample=0.75, colsample_bylevel=0.5, colsample_bytree=0.75,
                        n_estimators=95)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
xgboost.fit(x_train, y_train)
pred = xgboost.predict(x_test)
y_test = y_test.flatten()
print(accuracy_score(y_test, pred))
