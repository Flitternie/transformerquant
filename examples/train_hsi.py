#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from transformerquant.featurizers.default_featurizer import DefaultFeaturizer
from transformerquant.dataset.sampler import Sampler
from transformerquant.trainer.agent import Agent
from transformerquant.utils.datetime_converter import convert_str_to_dt
from transformerquant.configs.bert_config import BertConfig
from transformerquant.models.residual_bert import BertForPreTraining, BertForSequenceClassification

def create_feature_container(dropna=False):
    featurizer = DefaultFeaturizer(fwd_returns_window=1, task='regression')
    order_book_ids = ['^HSI']
    feature_container = {}
    for order_book_id in order_book_ids:
        try:
            print("process {}".format(order_book_id))
            data = yf.download(tickers=order_book_id, start='2010-01-01', end='2020-01-01').reset_index()
            data['Date'] = data['Date'].astype(str)
            open_yf = torch.tensor(data['Open'].values, dtype=torch.float32)
            high_yf = torch.tensor(data['High'].values, dtype=torch.float32)
            low_yf = torch.tensor(data['Low'].values, dtype=torch.float32)
            close_yf = torch.tensor(data['Close'].values, dtype=torch.float32) 
            volume_yf = torch.tensor(data['Volume'].values, dtype=torch.float32)
            #pdb.set_trace()
            output = featurizer.forward(open_yf,high_yf,low_yf,close_yf,volume_yf)
            data['datetime'] = data['Date'].apply(lambda x:convert_str_to_dt(x, format_="%Y-%m-%d"))
            output_np_list = [feature.cpu().detach().numpy() for feature in output]
            #pdb.set_trace()
            output_np = np.asarray(output_np_list).transpose(1,0)
            feature_df = pd.DataFrame(output_np, index=data['datetime'])
        except Exception as e:
            print("{} failed".format(order_book_id))
        else:
            print("{} successfully".format(order_book_id))
            if dropna:
                 feature_df = feature_df.dropna()
            #pdb.set_trace()
            feature_container[order_book_id] = feature_df
    return feature_container


def create_sample_container(feature_container, task='regression'):
    sequence_window = 30
    use_normalize = False
    frequency_x = '1d'
    batch_size = 32
    sampler = Sampler(sequence_window=sequence_window,
                      frequency_x=frequency_x,
                      interval_depart=False,
                      process_nan=True,
                      use_normalize = use_normalize,
                      saved_nomalizer_dir='/tmp/',
                      batch_size=batch_size,
                      train_ratio=0.7,
                      val_ratio = 0.1,
                      test_ratio = 0.2,
                      task = task)
    sample_container = sampler.generate_sample(feature_container)
    return sample_container


def create_model():
    config = BertConfig()
    config.d_model = 72
    model = BertForSequenceClassification(config)
    return model


def create_agent(model):
    use_cuda=True
    loss_func = torch.nn.MSELoss()
    n_epochs = 300
    lr = 0.001
    early_stop_patience = 80
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    to_save_dir = "../model/"
    checkpoint = None#"/media/allen/c54da21a-a3bc-4c5e-a36c-0a41b6108e59/quant/Quant_core/AI_Trader/asset/training_PositionModel_20180509_1814/model_PositionModel_282_val_loss=0.3601593.pth"
    #agent
    agent = Agent(model,
                  use_cuda=use_cuda,
                  loss_func=loss_func,
                  optimizer=optimizer,
                  lr_scheduler = lr_scheduler,
                  n_epochs=n_epochs,
                  early_stop_patience=early_stop_patience,
                  to_save_dir=to_save_dir,
                  checkpoint=checkpoint)
    return agent


def main(load=False):
    feature_container = create_feature_container(dropna=True)
    sample_container = create_sample_container(feature_container)
    model = create_model()
    agent = create_agent(model)
    state = agent.fit(sample_container['dataloader_train'], sample_container['dataloader_val'])
    return state

if __name__ == "__main__":
    main()