import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # For saving/loading models

class UpstreamViabilityAgent:
    """ Agent to predict collpase in cell viability during the upstream manufacturing process """
    def __init__(self, model_path="viability_collapse_rf.pkl"):
        self.model_path = model_path
        self.model = None

    def train(self, train_df):
        """ Method to train a ML model on historical data to predict cell viability collapse
            And saves the model to file .pkl """
        feat_cols = [
            'culture_time_hr', 'pH', 'temperature_C', 'viability_pct',
            'viable_cell_density_million_per_mL', 'dissolved_oxygen_pct',
            'carbon_dioxide_pct', 'titer_g_per_L', 'growth_rate_per_hr',
            'glucose_g_per_L', 'lactate_g_per_L', 'ammonia_mM', 'calcium_mM',
            'sodium_potassium_ratio', 'agitation_rpm', 'cell_bleed_volume_mL'
        ]
        X = train_df[feat_cols]
        y = train_df['viability_collapse']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        self.model = RandomForestClassifier() # can be swapped with other models such as XGBoost
        self.model.fit(X_train, y_train)
        print("Validation accuracy:", self.model.score(X_test, y_test))
        joblib.dump(self.model, self.model_path)

    def load(self):
        """ Method to load the model from file """
        self.model = joblib.load(self.model_path)

    def predict(self, sample_row):
        """ Method to predict the outcome for a given input sample """
        # sample_row: sample dataframe, e.g. one hourly sample
        feat_cols = [
            'culture_time_hr', 'pH', 'temperature_C', 'viability_pct',
            'viable_cell_density_million_per_mL', 'dissolved_oxygen_pct',
            'carbon_dioxide_pct', 'titer_g_per_L', 'growth_rate_per_hr',
            'glucose_g_per_L', 'lactate_g_per_L', 'ammonia_mM', 'calcium_mM',
            'sodium_potassium_ratio', 'agitation_rpm', 'cell_bleed_volume_mL'
        ]
        features = sample_row[feat_cols] #.values.reshape(1, -1)
        pred = self.model.predict(features)
        return int(pred[0])  # 1=collapse, 0=no collapse
