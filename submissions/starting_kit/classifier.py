from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self):
        # Definir les imputers et les scalers.
        qual_imputer = SimpleImputer(strategy="most_frequent")
        quant_imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()

        # Transformations.
        qual_transform = Pipeline(steps=[
            ("imputer", qual_imputer),
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
                ("qual", qual_transform, ['code_postal', 'code_insee_commune']),  
                ("quant", quant_transform, ['id', 'usr_diagnostiqueur_id', 'usr_logiciel_id', 'tr001_modele_dpe_id',
                                            'consommation_energie', 'estimation_ges', 'tr002_type_batiment_id',
                                            'tr012_categorie_erp_id', 'tr013_type_erp_id', 'annee_construction',
                                            'surface_habitable', 'surface_thermique_lot', 'tv016_departement_id',
                                            'portee_dpe_batiment', 'shon', 'surface_utile',
                                            'surface_thermique_parties_communes', 'en_souterrain', 'en_surface',
                                            'nombre_niveaux', 'nombre_circulations_verticales', 'nombre_boutiques',
                                            'presence_verriere', 'surface_verriere', 'type_vitrage_verriere',
                                            'nombre_entrees_avec_sas', 'nombre_entrees_sans_sas',
                                            'surface_baies_orientees_nord', 'surface_baies_orientees_est_ouest',
                                            'surface_baies_orientees_sud', 'surface_planchers_hauts_deperditifs',
                                            'surface_planchers_bas_deperditifs',
                                            'surface_parois_verticales_opaques_deperditives', 'etat_avancement',
                                            'dpe_vierge', 'est_efface']),  
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
