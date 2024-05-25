import pandas as pd

class AnomalyAnalizerModel:
    def __init__(self, model_path: str):
        self.response_data = pd.read_csv('/response_data.csv')
        self.model = joblib.load(model_path)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return self.model.predict(data)