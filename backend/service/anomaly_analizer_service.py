import service.anomaly_analyzer_model as anomaly_analyzer_model
import datetime as dt
from service.types import TimeSeriesType
from enum import Enum


class AnomalyAnalizerService:
    def __init__(self):
        self.model = anomaly_analyzer_model.AnomalyAnalizerModel()

    def get_series(self, series_type: TimeSeriesType):
        data = self.model.data[series_type]
        return data[['point', 'value']].to_dict(orient='records')

    def get_anomaly(self, series_type: TimeSeriesType, start_date: dt.datetime, end_date: dt.datetime):
        data = self.model.predict(series_type, start_date, end_date)
        return data