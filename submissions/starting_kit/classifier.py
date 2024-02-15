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
        X = X.astype(str)
        return X

class Classifier(BaseEstimator):
    def __init__(self):
        # Definir les imputers et les scalers.
        # Definir les imputers et les scalers.
        qual_imputer = SimpleImputer(strategy="most_frequent")
        quant_imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()

        # Transformations.

        # Transformations.
        qual_transform = Pipeline(steps=[
            ("previus", UniformingdType()),
            ("imputer", qual_imputer),
            ("encoder", OneHotEncoder(handle_unknown='ignore'))
        ])


        quant_transform = Pipeline(steps=[
            ("imputer", quant_imputer),
            ("scaler", scaler)
        ])

        # ColumnTransformer para apliquer des transformations
        # differentes a les columnes.

        # ColumnTransformer para apliquer des transformations
        # differentes a les columnes.
        self.transformer = ColumnTransformer(
            transformers=[
                ("qual", qual_transform, ['code_postal', 'code_insee_commune']),  
                ("quant", quant_transform, ['portee_dpe_batiment', 'shon', 'surface_utile',
                                            'surface_thermique_parties_communes', 'en_souterrain', 'en_surface',
                                            'nombre_niveaux', 'nombre_circulations_verticales',
                                            'type_vitrage_verriere', 'surface_baies_orientees_nord',
                                            'surface_baies_orientees_est_ouest', 'surface_baies_orientees_sud',
                                            'surface_planchers_hauts_deperditifs',
                                            'surface_planchers_bas_deperditifs',
                                            'surface_parois_verticales_opaques_deperditives', 'etat_avancement',
                                            'dpe_vierge', 'est_efface']),  
            ],
            remainder='drop'
        )

        # Definir le model de clasification.

        # Definir le model de clasification.
        self.model = LogisticRegression(max_iter=1000, random_state=42)

        # Construction du pipeline complet.

        # Construction du pipeline complet.
        self.pipe = make_pipeline(self.transformer, self.model)

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)