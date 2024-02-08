import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    def __init__(self):
        # Definir los imputers y scalers específicos para los atributos cualitativos y cuantitativos
        qual_imputer = SimpleImputer(strategy="most_frequent")
        quant_imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        
        # Definir las transformaciones para los atributos cualitativos y cuantitativos
        qual_transform = Pipeline(steps=[
            ("imputer", qual_imputer),
            ("encoder", OneHotEncoder(handle_unknown='ignore'))
        ])
        
        quant_transform = Pipeline(steps=[
            ("imputer", quant_imputer),
            ("scaler", scaler)
        ])
        
        # Definir el ColumnTransformer para aplicar transformaciones diferentes a diferentes tipos de columnas
        self.transformer = ColumnTransformer(
            transformers=[
                ("qual", qual_transform, ['code_postal', 'code_insee_commune']),  
                ("quant", quant_transform, self._get_numerical_columns())  
            ],
            remainder='drop'  
        )
        
        # Definir el modelo de clasificación
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Definir el pipeline completo
        self.pipe = make_pipeline(self.transformer, self.model)

    def _get_numerical_columns(self):
        """Devuelve las columnas numéricas del conjunto de datos."""
        # Aquí asumimos que X es un DataFrame de pandas
        return X.select_dtypes(include=np.number).columns.tolist()

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)
