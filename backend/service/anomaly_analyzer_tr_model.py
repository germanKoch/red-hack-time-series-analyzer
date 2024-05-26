import os
import torch
import torch.nn as nn
import pandas as pd
import math
# import dgl
# from dgl.nn import GATConv
from time import time
from torch.nn import TransformerEncoder
from backend.service.types import TimeSeriesType
from torch.nn import TransformerDecoder
import numpy as np
import datetime as dt

from torch.utils.data import Dataset, DataLoader, TensorDataset

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
    return model

def min_max_scaling(df, phi=0.05):
    df_normalized = df.copy()
    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df_normalized[column] = (df[column] - min_val) / (max_val - min_val + phi)
    return df_normalized

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)

def get_inference(model, data, dataO):
    torch.zero_grad = True
    model.eval()
    feats = dataO.shape[1]
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.DoubleTensor(data)
    dataset = TensorDataset(data_x, data_x)
    bs = len(data)
    dataloader = DataLoader(dataset, batch_size=bs)
    l1s, l2s = [], []
    for d, _ in dataloader:
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, bs, feats)
        z = model(window, elem)
        if isinstance(z, tuple):
            z = z[1]
    loss = l(z, elem)[0]
    return loss.detach().numpy(), z.detach().numpy()[0]

def predict_loss(model, data):
    timestamp = data["timestamp"]
    data = data[["web_response", "throughput", "error", "apdex"]]
    test_norm = min_max_scaling(data, phi=0.0)
    test_loader = DataLoader(np.array(test_norm), batch_size=len(test_norm))

    testD = next(iter(test_loader))
    testO = testD

    testD = convert_to_windows(testD, model)
    result = get_inference(model, testD, testO)
    loss = pd.DataFrame(result[0], columns=['web_response', 'throughput', 'error', 'apdex'])
    loss['timestamp'] = timestamp
    return loss

class AnomalyAnalizerModel:
    def __init__(self):
        ROOT = os.path.dirname(os.path.abspath(__file__))
        resources = os.path.join(ROOT, "..", "resources")
        model = load_checkpoint(os.path.join(resources, 'models', 'model.ckpt'))
        
        self.data = self._load_df(
            os.path.join(resources, 'datasets', 'web_response.csv'), 
            os.path.join(resources, 'datasets', 'throughput.csv'), 
            os.path.join(resources, 'datasets', 'error.csv'), 
            os.path.join(resources, 'datasets', 'apdex.csv')
        )
        self.loss = predict_loss(model, self.data)
        self.quantiles = self.loss[["web_response", "throughput", "error", "apdex"]].quantile(0.99).to_dict()
        
    def _load_df(self, RESP_DATA, THR_DATA, ERROR_DATA, APDEX_DATA):
        web_response = pd.read_csv(RESP_DATA, header=0, sep=',', index_col=0).reset_index()
        throughput = pd.read_csv(THR_DATA, header=0, sep=',', index_col=0).reset_index()
        error = pd.read_csv(ERROR_DATA, header=0, sep=',', index_col=0).reset_index()
        apdex = pd.read_csv(APDEX_DATA, header=0, sep=',', index_col=0).reset_index()
        
        apdex.rename(columns={"point": "timestamp", "value": "apdex"}, inplace=True)
        web_response.rename(columns={"point": "timestamp", "value": "web_response"}, inplace=True)
        throughput.rename(columns={"point": "timestamp", "value": "throughput"}, inplace=True)
        error.rename(columns={"point": "timestamp", "value": "error"}, inplace=True)
        
        ts_df = pd.merge(web_response, throughput, on="timestamp")
        ts_df = pd.merge(ts_df, error, on="timestamp")
        ts_df = pd.merge(ts_df, apdex, on="timestamp")
        ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
        return ts_df    
    
    def get_data(self, series_type: TimeSeriesType):
        if series_type == TimeSeriesType.RESPONSE:
            return self.data[['timestamp', 'web_response']].rename(columns={'web_response': 'value', 'timestamp': 'point'})
        elif series_type == TimeSeriesType.THROUGHPUT:
            return self.data[['timestamp', 'throughput']].rename(columns={'throughput': 'value', 'timestamp': 'point'})
        elif series_type == TimeSeriesType.APDEX:
            return self.data[['timestamp', 'apdex']].rename(columns={'apdex': 'value', 'timestamp': 'point'})
        elif series_type == TimeSeriesType.ERROR:
            return self.data[['timestamp', 'error']].rename(columns={'error': 'value', 'timestamp': 'point'})

    def predict(self, series_type: TimeSeriesType, start_date: dt.datetime, end_date: dt.datetime) -> pd.Series:
        # iterate over the winddows with len 128 between start_date and end_date
        def get_anomaly(self, quantile, column, start_date: dt.datetime, end_date: dt.datetime):
            filtered = self.loss[(self.loss['timestamp'] >= start_date) & (self.loss['timestamp'] <= end_date)]
            filtered = filtered[filtered[column] > quantile]
            return [t.to_pydatetime() for t in filtered['timestamp']]

        if series_type == TimeSeriesType.RESPONSE:
            return get_anomaly(self, self.quantiles['web_response'], 'web_response', start_date, end_date)
        elif series_type == TimeSeriesType.THROUGHPUT:
            return get_anomaly(self, self.quantiles['throughput'], 'throughput', start_date, end_date)
        elif series_type == TimeSeriesType.APDEX:
            return get_anomaly(self, self.quantiles['apdex'], 'apdex', start_date, end_date)
        elif series_type == TimeSeriesType.ERROR:
            return get_anomaly(self, self.quantiles['error'], 'error', start_date, end_date)