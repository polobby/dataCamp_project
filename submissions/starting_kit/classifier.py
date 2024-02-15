from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class UniformingdType(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.astype(object)
        return X

class toStr(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.astype(str)
        return X


class Classifier(BaseEstimator):
    def __init__(self):
        # Definir les imputers et les scalers.
        qual_imputer = SimpleImputer(strategy="most_frequent")
        quant_imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()

        # Transformations.
        qual_transform = Pipeline(steps=[
            ("previus", UniformingdType()),
            ("imputer", qual_imputer),
            ("toStr", toStr()),
            ("encoder", OneHotEncoder(handle_unknown='ignore'))
        ])

        quant_transform = Pipeline(steps=[
            ("imputer", quant_imputer),
            ("scaler", scaler)
        ])

        # ColumnTransformer para apliquer des transformations
        # differentes a les columnes.
        self.transformer = ColumnTransformer(
            transformers=[
                ("qual", qual_transform, [19, 20]),  
                ("quant", quant_transform, [23, 24, 25, 26, 27, 28, 29, 30, 34, 37, 38, 39, 40, 41, 42, 43, 45, 46]),  
            ],
            remainder='drop'  
        )

        # Definir le model de clasification.
        self.model = LogisticRegression(max_iter=1000, random_state=42)

        # Construction du pipeline complet.
        self.pipe = make_pipeline(self.transformer, self.model)

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)