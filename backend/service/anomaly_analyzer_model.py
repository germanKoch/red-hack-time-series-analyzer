import pandas as pd
import os
from torch import nn
import torch
import math
import datetime as dt
from backend.service.types import TimeSeriesType
from statsmodels.tsa.seasonal import seasonal_decompose

class ConvAutoencoder(nn.Module):
    def __init__(self, loss, hidden_size, seq_len, dropout_prob):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_size, 3, padding=1),  # Output: [batch_size, 32, seq_len]
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),       # Output: [batch_size, 32, seq_len//2]
            nn.Conv1d(hidden_size, hidden_size//2, 3, padding=1), # Output: [batch_size, 16, seq_len//2]
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),       # Output: [batch_size, 16, seq_len//4]
            nn.Conv1d(hidden_size//2, hidden_size//4, 3, padding=1),  # Output: [batch_size, 8, seq_len//4]
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)        # Output: [batch_size, 8, seq_len//8]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_size//4, hidden_size//2, 4, stride=2, padding=1), # Output: [batch_size, 16, seq_len//4]
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size//2, hidden_size, 4, stride=2, padding=1), # Output: [batch_size, 32, seq_len//2]
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size, 1, 4, stride=2, padding=1),  # Output: [batch_size, 1, seq_len]
        )
        self.loss = loss
        self.seq_len = seq_len
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 1] -> [batch_size, 1, seq_len]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.permute(0, 2, 1)  # [batch_size, 1, seq_len] -> [batch_size, seq_len, 1]
        return decoded
    
def load_checkpoint(path):
    model = ConvAutoencoder(nn.MSELoss(), 128, 128, 0.2)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.threshold = checkpoint['threshold']
    model.scaler = checkpoint['scaler']
    model.seq_len = checkpoint.get('seq_len', 128)
    return model

class AnomalyAnalizerModel:
    def __init__(self):
        ROOT = os.path.dirname(os.path.abspath(__file__))
        resources = os.path.join(ROOT, "..", "resources")
        self.response_model = load_checkpoint(os.path.join(resources, 'models', 'response_model.pth'))
        self.throughput_model = load_checkpoint(os.path.join(resources, 'models', 'throughput_model.pth'))
        self.apdex_model = load_checkpoint(os.path.join(resources, 'models', 'apdex_model.pth'))

        self.response_data = self._load_df(os.path.join(resources, 'datasets', 'web_response.csv'), self.response_model.scaler)
        self.apdex_data = self._load_df(os.path.join(resources, 'datasets', 'apdex.csv'), self.apdex_model.scaler)
        self.thoughput_data = self._load_df(os.path.join(resources, 'datasets', 'throughput.csv'), self.throughput_model.scaler)

        self.models = {
            TimeSeriesType.RESPONSE: self.response_model,
            TimeSeriesType.APDEX: self.apdex_model,
            TimeSeriesType.THROUGHPUT: self.throughput_model
        }
        self.data = {
            TimeSeriesType.RESPONSE: self.response_data,
            TimeSeriesType.APDEX: self.apdex_data,
            TimeSeriesType.THROUGHPUT: self.thoughput_data
        }

    def _load_df(self, path, scaler):
        data = pd.read_csv(path, parse_dates=['point'])
        data['point'] = pd.to_datetime(data['point'])
        data.set_index('point', inplace=True)
        result = seasonal_decompose(data['value'], model='additive', period=1440)  # Period is the number of minutes in a day
        data['de_seasonalized'] = data['value'] - result.seasonal
        data.reset_index(inplace=True)
        print(scaler)
        data['normilized'] = scaler.transform(data['de_seasonalized'].values.reshape(-1, 1))
        return data    

    def predict(self, series_type: TimeSeriesType, start_date: dt.datetime, end_date: dt.datetime) -> pd.Series:
        # iterate over the winddows with len 128 between start_date and end_date
        model = self.models[series_type]
        data = self.data[series_type]

        diff = end_date - start_date
        diff_minutes = math.ceil(((diff.days * 24 * 60) + (diff.seconds/60))/128)*128

        data = data[(data['point'] >= start_date)].head(diff_minutes)

        def detect_anomalies(model, time_series, threshold, window_size):
            model.eval()  # Set the model to evaluation mode
            anomalies = []
            predicted_values = []
            print('ITERATIONS', len(time_series) - window_size + 1)
            with torch.no_grad():  # Disable gradient calculation
                for i in range(0, len(time_series) - window_size + 1, window_size):
                    window = time_series[i:i + window_size].unsqueeze(0).unsqueeze(2)  # Add batch dimension
                    prediction = model(window).squeeze(0)
                    predicted_values.extend(prediction.tolist())

                    error = torch.abs(prediction - window.squeeze(0))
                    for j in range(window_size):
                        if error[j].item() > threshold:
                            anomalies.append(i + j)  # Mark the specific point as an anomaly
            return anomalies, predicted_values

        deseasonalized = torch.FloatTensor(data['normilized'].values)
        anomalies, predicted_values = detect_anomalies(model, deseasonalized, model.threshold, model.seq_len)
        result = [t.to_pydatetime() for t in data.iloc[anomalies].point.to_list()]
        result = [t for t in result if t <= end_date]
        print(result)
        print(end_date)
        return result