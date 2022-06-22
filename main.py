# %% import statements
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# %% division function
def weird_division(n, d):
    return n / d if d else 0.0


# %% Read data in
data = pd.read_csv("data/nba_logreg.csv")
data.rename(columns={'3P Made': '3PM'}, inplace=True)
data[['TARGET_5Yrs']] = data[['TARGET_5Yrs']].astype(int).astype(str)

# %% EDA

print(data.head())

print(data.info())  # - from this I noticed that 3P% is missing 11 values
print(data.isnull().sum())

# %% Data Imputation
# Let's impute the missing values. We know that 3P% is 3PM / 3PA * 100

data['3P%'] = data.apply(lambda row: (
    weird_division(row['3PM'], row['3PA'])) * 100, axis=1)

# Creating some interesting graphs

# %% Create a correlation matrix for all the variables
corr = data[['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%',
            'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
             'TOV']].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# %% Compare some of the highly correlated variables split by the target variable
fig, axes = plt.subplots(1, 3, figsize=(15, 15))
fig.suptitle("Comparision of metrics split by 5 Years")

sns.scatterplot(ax=axes[0], x="PTS", y="MIN", hue="TARGET_5Yrs", data=data)
axes[0].set_title("Points vs Minutes")

sns.scatterplot(ax=axes[1], x="AST", y="TOV", hue="TARGET_5Yrs", data=data)
axes[1].set_title("Assist vs TurnOvers")

sns.scatterplot(ax=axes[2], x="REB", y="BLK", hue="TARGET_5Yrs", data=data)
axes[2].set_title("Rebounds vs Blocks")

plt.show()

# Now let's split the data into test and training and build some models.

# %% Split the data
