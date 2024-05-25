import backend.service.anomaly_analyzer_model as anomaly_analyzer_model
import datetime as dt
from backend.service.types import TimeSeriesType
from enum import Enum


class AnomalyAnalizerService:
    def __init__(self):
        self.model = anomaly_analyzer_model.AnomalyAnalizerModel()

    def get_series(self, series_type: TimeSeriesType):
        data = self.model.data[series_type]
        series = data[['point', 'value']].to_dict(orient='records')
        # Convert the datetime objects to string
        for record in series:
            record['point'] = record['point'].isoformat()
        return series

    def get_anomaly(self, series_type: TimeSeriesType, start_date: dt.datetime, end_date: dt.datetime):
        data = self.model.predict(series_type, start_date, end_date)
        return data