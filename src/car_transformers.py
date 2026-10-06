import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BrandTargetEncoder(BaseEstimator, TransformerMixin):
    """Replace `brand` by the mean target value of that brand (fit on training data only).

    Output keeps all other columns and appends `brand_encoded` as the last column
    (same layout as the original notebook).
    """

    def __init__(self, column="brand", output_column="brand_encoded"):
        self.column = column
        self.output_column = output_column

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y, dtype=float))
        self.mapping_ = y.groupby(X[self.column].to_numpy()).mean()
        self.global_mean_ = float(y.mean())
        self.columns_out_ = [c for c in X.columns if c != self.column] + [self.output_column]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.output_column] = X[self.column].map(self.mapping_).fillna(self.global_mean_)
        return X[self.columns_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_out_, dtype=object)
