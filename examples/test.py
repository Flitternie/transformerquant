import numpy as np
import pandas as pd
import torch
import yfinance as yf
from transformerquant.featurizers.default_featurizer import DefaultFeaturizer
from transformerquant.dataset.sampler import Sampler

featurizer = DefaultFeaturizer(fwd_returns_window=1, task='regression')
data = yf.download(tickers="^HSI", start='2010-01-01', end='2020-01-01').reset_index()
open_yf = torch.tensor(data['Open'].values, dtype=torch.float32)
high_yf = torch.tensor(data['High'].values, dtype=torch.float32)
low_yf = torch.tensor(data['Low'].values, dtype=torch.float32)
close_yf =torch.tensor(data['Close'].values, dtype=torch.float32) 
volume_yf = torch.tensor(data['Volume'].values, dtype=torch.float32)
#pdb.set_trace()
output = featurizer.forward(open_yf,high_yf,low_yf,close_yf,volume_yf)
data['datetime'] = data['Date'].apply(lambda x:convert_str_to_dt(x, format_="%Y-%m-%d"))
output_np_list = [feature.cpu().detach().numpy() for feature in output]
#pdb.set_trace()
output_np = np.asarray(output_np_list).transpose(1,0)
feature_df = pd.DataFrame(output_np, index=data['datetime'])