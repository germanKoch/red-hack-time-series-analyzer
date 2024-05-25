import pandas as pd
import os
from torch import nn
import torch
import math
import datetime as dt
from backend.service.types import TimeSeriesType
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from statsmodels.tsa.seasonal import seasonal_decompose

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None, is_causal=True):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            tgt_is_causal=True,
            memory_is_causal=True
        ):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TranAD(nn.Module):
    def __init__(self, feats, lr, batch_size, window_size):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = batch_size
        self.n_feats = feats
        self.n_window = window_size
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
        self.double()

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

def load_checkpoint(path):
    BATCH_SIZE = 64
    NUM_LABELS = 4
    LEARNING_RATE = 10e-5
    WINDOW_SIZE = 20
    model = TranAD(
        feats=NUM_LABELS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE
    )
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

class AnomalyAnalizerModel:
    def __init__(self):
        ROOT = os.path.dirname(os.path.abspath(__file__))
        resources = os.path.join(ROOT, "..", "resources")
        self.model = load_checkpoint(os.path.join(resources, 'models', 'model.ckpt'))
        self.data = self._load_df(
            os.path.join(resources, 'datasets', 'response.csv'), 
            os.path.join(resources, 'datasets', 'throughput.csv'), 
            os.path.join(resources, 'datasets', 'error.csv'), 
            os.path.join(resources, 'datasets', 'apdex.csv')
        )
        
    def _load_df(self, RESP_DATA, THR_DATA, ERROR_DATA, APDEX_DATA):
        web_response = pd.read_csv(RESP_DATA, header=0, sep=',', index_col=0).reset_index()
        throughput = pd.read_csv(THR_DATA, header=0, sep=',', index_col=0).reset_index()
        error = pd.read_csv(ERROR_DATA, header=0, sep=',', index_col=0).reset_index()
        apdex = pd.read_csv(APDEX_DATA, header=0, sep=',', index_col=0).reset_index()
        web_response.rename(columns={"point": "timestamp"}, inplace=True)
        throughput.rename(columns={"point": "timestamp", "sum_call_count": "throughput"}, inplace=True)
        error.rename(columns={"point": "timestamp", "ratio": "error"}, inplace=True)
        ts_df = pd.merge(web_response, throughput, on="timestamp")
        ts_df = pd.merge(ts_df, error, on="timestamp")
        ts_df = pd.merge(ts_df, apdex, on="timestamp")
        ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
        return ts_df    

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