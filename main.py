import pandas as pd
from genetic_feature_selection import run_feature_selection_from_df

df = pd.read_csv('../data/wine.csv')
selected_features, classes = run_feature_selection_from_df(df,
                                                           target_col='Type',
                                                           classifier = 'SVM',
                                                           n_gen=10)
print(selected_features)
