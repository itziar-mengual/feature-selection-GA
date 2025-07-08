from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

def run_elastic_net_selection(df: pd.DataFrame, feature_cols: list, target_col: str):
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df[target_col].values)

    # Pipeline con escalado + modelo con validación cruzada
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegressionCV(
            penalty='elasticnet', solver='saga', l1_ratios=[0.5],
            cv=5, scoring='accuracy', max_iter=5000, random_state=42
        ))
    ])
    pipeline.fit(X, y)
    model = pipeline.named_steps['clf']
    coef = model.coef_[0]
    selected = [feature_cols[i] for i, c in enumerate(coef) if abs(c) > 1e-4]

    return selected, model.score(X, y)
