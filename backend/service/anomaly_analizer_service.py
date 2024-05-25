class AnomalyAnalizerService:
    def __init__(self):
        self.anomaly_analizer = AnomalyAnalizer()

    def get_anomaly(self, data):
        return self.anomaly_analizer.get_anomaly(data)