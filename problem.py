# Importation des modules nécessaires
import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

# Titre
problem_title = "Energetic class prediction for housing in Paris"
_prediction_label_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Correspondance entre catégoriel et int8
int_to_cat = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
}
# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}
_prediction_label_int = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_int)

# Implémentation du workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(name='balanced acc'),
    rw.score_types.Accuracy(name="acc", precision=3)
    ]

# Construction des données
# _change_column_names = {'date_visite_diagnostiqueur':'datetime64[ns]', 
#                         'date_etablissement_dpe':'datetime64[ns]',
#                         'date_arrete_tarifs_energies':'datetime64[ns]',
#                         'code_postal':'string',
#                         'code_insee_commune':'string',
#                         'code_insee_commune_actualise':'string',
#                         'date_reception_dpe':'datetime64[ns]'}
_target_column_name = 'classe_consommation_energie'
_ignore_column_names = ['id', 'numero_dpe', 'version_methode_dpe',
       'nom_methode_etude_thermique', 'version_methode_etude_thermique',
       'commentaires_ameliorations_recommandations',
       'explication_personnalisee', 'estimation_ges',
       'classe_estimation_ges', 'nom_rue', 'numero_rue',
       'batiment', 'escalier', 'etage', 'porte', 'numero_lot',
       'partie_batiment', 'adresse_organisme_certificateur']


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))

    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    # X_df = X_df.astype(_change_column_names)

    y_array = data[_target_column_name]
    y_array = y_array.map(cat_to_int).fillna(-1).astype("int8").values
    # X_df as an np.array
    X = X_df.values
    return X, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    return cv.split(X, y)
