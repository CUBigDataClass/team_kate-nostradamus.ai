# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from datetime import datetime
import time
import json

from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, event,exc
from collections import defaultdict

import pandas as pd


from apps import db,scheduler

import requests



class BTC(db.Model):

    
    __tablename__ = 'btc'
    time = db.Column(db.Float(), primary_key=True)
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    open = db.Column(db.Float())
    close = db.Column(db.Float())
    volumeto = db.Column(db.Float())
    volumefrom = db.Column(db.Float())
    
    @classmethod
    def get_historical(self):
        return self.query.all()
    
    @classmethod
    def get_by_time(cls, time):
        return cls.query.filter_by(time=time).first()
    

    def toJSON(self):
        return self.toDICT()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

class BTC_forecasts(db.Model):

    __tablename__ = 'btc_predictions'
    time = db.Column(db.Float(), primary_key=True)
    close_1 = db.Column(db.Float())
    close_2 = db.Column(db.Float())
    close_3 = db.Column(db.Float())
    close_4 = db.Column(db.Float())
    close_5 = db.Column(db.Float())
    close_6 = db.Column(db.Float())
    close_7 = db.Column(db.Float())
    close_8 = db.Column(db.Float())
    close_9 = db.Column(db.Float())
    close_10 = db.Column(db.Float())
    close_11 = db.Column(db.Float())
    close_12 = db.Column(db.Float())
    close_13 = db.Column(db.Float())
    close_14 = db.Column(db.Float())

    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


@event.listens_for(BTC.__table__, 'after_create')
def create_BTC(*args, **kwargs):

    api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
    btcUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&allData=true&'
    resBTC = requests.get(btcUrl+api_key).json()['Data']['Data']

    for days in resBTC:
        if days['low']>0:
            row=BTC(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)
            db.session.commit()

@event.listens_for(BTC_forecasts.__table__, 'after_create')
def create_BTC_forecasts(*args, **kwargs):

    btc=BTC.query.all()
    btc=pd.DataFrame(BTC.toDICT(btc))
    btc_preds = get_predictions(btc)
    curr_date = datetime.now()
    curr_time = time.mktime(curr_date.timetuple())
    
    row=BTC_forecasts(time=curr_time, close_1=btc_preds[0], close_2=btc_preds[1], close_3=btc_preds[2], close_4=btc_preds[3], close_5=btc_preds[4], close_6=btc_preds[5], close_7=btc_preds[6], close_8=btc_preds[7], close_9=btc_preds[8], close_10=btc_preds[9], close_11=btc_preds[10], close_12=btc_preds[11], close_13=btc_preds[12], close_14=btc_preds[13])
    
    db.session.add(row)
    db.session.commit()




class ETH(db.Model):
    __tablename__ = 'eth'
    time = db.Column(db.Float(), primary_key=True)
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    open = db.Column(db.Float())
    close = db.Column(db.Float())
    volumeto = db.Column(db.Float())
    volumefrom = db.Column(db.Float())
    
    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


class ETH_forecasts(db.Model):

    __tablename__ = 'eth_predictions'
    time = db.Column(db.Float(), primary_key=True)
    close_1 = db.Column(db.Float())
    close_2 = db.Column(db.Float())
    close_3 = db.Column(db.Float())
    close_4 = db.Column(db.Float())
    close_5 = db.Column(db.Float())
    close_6 = db.Column(db.Float())
    close_7 = db.Column(db.Float())
    close_8 = db.Column(db.Float())
    close_9 = db.Column(db.Float())
    close_10 = db.Column(db.Float())
    close_11 = db.Column(db.Float())
    close_12 = db.Column(db.Float())
    close_13 = db.Column(db.Float())
    close_14 = db.Column(db.Float())

    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


   

@event.listens_for(ETH.__table__, 'after_create')
def create_ETH(*args, **kwargs):

    api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
    ethUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&allData=true&'
    resETH = requests.get(ethUrl+api_key).json()['Data']['Data']

    for days in resETH:
        if days['low']>0:
            row=ETH(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)
            db.session.commit()


@event.listens_for(ETH_forecasts.__table__, 'after_create')
def create_ETH_forecasts(*args, **kwargs):

    eth=ETH.query.all()
    eth=pd.DataFrame(ETH.toDICT(eth))

    eth_preds = get_predictions(eth)
    curr_date = datetime.now()
    curr_time = time.mktime(curr_date.timetuple())
    
    row=ETH_forecasts(time=curr_time, close_1=eth_preds[0], close_2=eth_preds[1], close_3=eth_preds[2], close_4=eth_preds[3], close_5=eth_preds[4], close_6=eth_preds[5], close_7=eth_preds[6], close_8=eth_preds[7], close_9=eth_preds[8], close_10=eth_preds[9], close_11=eth_preds[10], close_12=eth_preds[11], close_13=eth_preds[12], close_14=eth_preds[13])
    
    db.session.add(row)
    db.session.commit()


class XMR(db.Model):
    __tablename__ = 'xmr'
    time = db.Column(db.Float(), primary_key=True)
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    open = db.Column(db.Float())
    close = db.Column(db.Float())
    volumeto = db.Column(db.Float())
    volumefrom = db.Column(db.Float())
    
    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


class XMR_forecasts(db.Model):

    __tablename__ = 'xmr_predictions'

    time = db.Column(db.Float(), primary_key=True)
    close_1 = db.Column(db.Float())
    close_2 = db.Column(db.Float())
    close_3 = db.Column(db.Float())
    close_4 = db.Column(db.Float())
    close_5 = db.Column(db.Float())
    close_6 = db.Column(db.Float())
    close_7 = db.Column(db.Float())
    close_8 = db.Column(db.Float())
    close_9 = db.Column(db.Float())
    close_10 = db.Column(db.Float())
    close_11 = db.Column(db.Float())
    close_12 = db.Column(db.Float())
    close_13 = db.Column(db.Float())
    close_14 = db.Column(db.Float())

    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


@event.listens_for(XMR.__table__, 'after_create')
def create_XMR(*args, **kwargs):

    api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
    xmrUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=XMR&tsym=USD&allData=true&'
    resXMR = requests.get(xmrUrl+api_key).json()['Data']['Data']

    for days in resXMR:
        if days['low']>0:
            row=XMR(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)
            db.session.commit()



@event.listens_for(XMR_forecasts.__table__, 'after_create')
def create_XMR_forecasts(*args, **kwargs):

    xmr=XMR.query.all()
    xmr=pd.DataFrame(XMR.toDICT(xmr))

    xmr_preds = get_predictions(xmr)
    curr_date = datetime.now()
    curr_time = time.mktime(curr_date.timetuple())
    
    row=XMR_forecasts(time=curr_time, close_1=xmr_preds[0], close_2=xmr_preds[1], close_3=xmr_preds[2], close_4=xmr_preds[3], close_5=xmr_preds[4], close_6=xmr_preds[5], close_7=xmr_preds[6], close_8=xmr_preds[7], close_9=xmr_preds[8], close_10=xmr_preds[9], close_11=xmr_preds[10], close_12=xmr_preds[11], close_13=xmr_preds[12], close_14=xmr_preds[13])
    db.session.add(row)
    db.session.commit()

# <<<<<<< Updated upstream
# # @scheduler.task('interval', id='update_daily_values', seconds=5)
# # def daily_db_update():
# =======
@scheduler.task('interval', id='update_daily_values', hours=24)
def daily_db_update():
    with scheduler.app.app_context():
        
        api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
        btcUrl_day='https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=1&'
        resBTC = requests.get(btcUrl_day+api_key).json()['Data']['Data']

        day_btc = resBTC[1]

        row1=BTC(time=day_btc['time'],high=day_btc['high'],low=day_btc['low'],open=day_btc['open'],close=day_btc['close'],volumeto=day_btc['volumeto'],volumefrom=day_btc['volumefrom'])
        db.session.add(row1)

        try:
            db.session.commit()
        except exc.SQLAlchemyError:
            pass

        ethUrl_day='https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&&limit=1&'
        resETH = requests.get(ethUrl_day+api_key).json()['Data']['Data']

        day_eth = resETH[1]
       
        row2=ETH(time=day_eth['time'],high=day_eth['high'],low=day_eth['low'],open=day_eth['open'],close=day_eth['close'],volumeto=day_eth['volumeto'],volumefrom=day_eth['volumefrom'])
        db.session.add(row2)

        try:
            db.session.commit()
        except exc.SQLAlchemyError:
            pass 
       

        xmrUrl_day='https://min-api.cryptocompare.com/data/v2/histoday?fsym=XMR&tsym=USD&&limit=1&'
        resXMR = requests.get(xmrUrl_day+api_key).json()['Data']['Data']

        day_xmr = resXMR[1]
        
        row3=XMR(time=day_xmr['time'],high=day_xmr['high'],low=day_xmr['low'],open=day_xmr['open'],close=day_xmr['close'],volumeto=day_xmr['volumeto'],volumefrom=day_xmr['volumefrom'])
        db.session.add(row3)
        try:
            db.session.commit()
        except exc.SQLAlchemyError:
            pass 


@scheduler.task('interval', id='make_biweekly_predictions', weeks=2)
def biweekly_db_update():

    with scheduler.app.app_context():

        curr_date = datetime.now()
        curr_time = time.mktime(curr_date.timetuple())

        btc=BTC.query.all()
        btc=pd.DataFrame(BTC.toDICT(btc))
        btc_preds = get_predictions(btc)
        row1=BTC_forecasts(time=curr_time, close_1=btc_preds[0], close_2=btc_preds[1], close_3=btc_preds[2], close_4=btc_preds[3], close_5=btc_preds[4], close_6=btc_preds[5], close_7=btc_preds[6], close_8=btc_preds[7], close_9=btc_preds[8], close_10=btc_preds[9], close_11=btc_preds[10], close_12=btc_preds[11], close_13=btc_preds[12], close_14=btc_preds[13])
        db.session.add(row1)
        db.session.commit()
        
        eth=ETH.query.all()
        eth=pd.DataFrame(ETH.toDICT(eth))
        eth_preds = get_predictions(eth)
        row2=ETH_forecasts(time=curr_time, close_1=eth_preds[0], close_2=eth_preds[1], close_3=eth_preds[2], close_4=eth_preds[3], close_5=eth_preds[4], close_6=eth_preds[5], close_7=eth_preds[6], close_8=eth_preds[7], close_9=eth_preds[8], close_10=eth_preds[9], close_11=eth_preds[10], close_12=eth_preds[11], close_13=eth_preds[12], close_14=eth_preds[13])
        db.session.add(row2)
        db.session.commit()

        xmr=XMR.query.all()
        xmr=pd.DataFrame(XMR.toDICT(xmr))
        xmr_preds = get_predictions(xmr)
        row3=XMR_forecasts(time=curr_time, close_1=xmr_preds[0], close_2=xmr_preds[1], close_3=xmr_preds[2], close_4=xmr_preds[3], close_5=xmr_preds[4], close_6=xmr_preds[5], close_7=xmr_preds[6], close_8=xmr_preds[7], close_9=xmr_preds[8], close_10=xmr_preds[9], close_11=xmr_preds[10], close_12=xmr_preds[11], close_13=xmr_preds[12], close_14=xmr_preds[13])
        db.session.add(row3)
        db.session.commit()

def toDICT(rset):
    result = defaultdict(list)
    for obj in rset:
        instance = inspect(obj)
        for key, x in instance.attrs.items():
            result[key].append(x.value)
    return result
    
def get_crypto():
    btc=BTC.query.all()
    eth=ETH.query.all()
    xmr=XMR.query.all()
    btc=pd.DataFrame(BTC.toDICT(btc))
    eth=pd.DataFrame(ETH.toDICT(eth))
    xmr=pd.DataFrame(XMR.toDICT(xmr))
    data=[btc,eth,xmr]

    return data
    
def get_eth_data():

    eth=ETH.query.all()
    eth_pred=ETH_forecasts.query.all()
    
    eth_df = pd.DataFrame(ETH.toDICT(eth))
    eth_pred= pd.DataFrame(ETH.toDICT(eth_pred))
    
    
    start = len(eth_df)-70

    eth_df = eth_df[['close']]
    eth_actual = (eth_df.iloc[start:,].values.ravel()).tolist()

    idx_actual = [i for i in range(0, len(eth_actual))]
    json_actuals = []

    for actual,idx in zip(eth_actual, idx_actual):

        json_actuals.append({'x': idx, 'y': actual })
    
    
    
    idx_pred = [i for i in range(len(eth_actual), len(eth_actual)+14)]
    json_preds = []
    json_preds.append(json_actuals[-1])
    
    close1 = eth_pred.close_1.values
    close2 = eth_pred.close_2.values
    close3 = eth_pred.close_3.values
    close4 = eth_pred.close_4.values
    close5 = eth_pred.close_5.values
    close6 = eth_pred.close_6.values
    close7 = eth_pred.close_7.values
    close8 = eth_pred.close_8.values
    close9 = eth_pred.close_9.values
    close10 = eth_pred.close_10.values
    close11 = eth_pred.close_11.values
    close12 = eth_pred.close_12.values
    close13 = eth_pred.close_13.values
    close14 = eth_pred.close_14.values

    preds = [close1[0],close2[0], close3[0], close4[0], close5[0], close6[0], close7[0], close8[0], close9[0], close10[0], close11[0], close12[0], close13[0], close14[0]]
    
    for pred,idx in zip(preds, idx_pred):

        json_preds.append({'x':idx ,'y':pred})

    print(json_actuals)
    print(json_preds)
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_btc_data():

    btc=BTC.query.all()
    btc_pred=BTC_forecasts.query.all()


    btc_df = pd.DataFrame(BTC.toDICT(btc))
    btc_pred = pd.DataFrame(ETH.toDICT(btc_pred))

    start = len(btc_df)-70

    btc_df = btc_df[['close']]
    btc_actual = (btc_df.iloc[start:,].values.ravel()).tolist()

    idx_actual = [i for i in range(0, len(btc_actual))]
    json_actuals = []

    for actual,idx in zip(btc_actual, idx_actual):

        json_actuals.append({'x': idx, 'y': actual })
    
    
    
    idx_pred = [i for i in range(len(btc_actual), len(btc_actual)+14)]
    json_preds = []
    json_preds.append(json_actuals[-1])
    

    close1 = btc_pred.close_1.values
    close2 = btc_pred.close_2.values
    close3 = btc_pred.close_3.values
    close4 = btc_pred.close_4.values
    close5 = btc_pred.close_5.values
    close6 = btc_pred.close_6.values
    close7 = btc_pred.close_7.values
    close8 = btc_pred.close_8.values
    close9 = btc_pred.close_9.values
    close10 = btc_pred.close_10.values
    close11 = btc_pred.close_11.values
    close12 = btc_pred.close_12.values
    close13 = btc_pred.close_13.values
    close14 = btc_pred.close_14.values

    preds = [close1[0],close2[0], close3[0], close4[0], close5[0], close6[0], close7[0], close8[0], close9[0], close10[0], close11[0], close12[0], close13[0], close14[0]]
    
    for pred,idx in zip(preds, idx_pred):

        json_preds.append({'x':idx ,'y':pred})

    
  
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_xmr_data():

    xmr=XMR.query.all()
    xmr_pred=XMR_forecasts.query.all()

    xmr_df = pd.DataFrame(XMR.toDICT(xmr))
    xmr_pred = pd.DataFrame(ETH.toDICT(xmr_pred))


    start = len(xmr_df)-70

    xmr_df =  xmr_df[['close']]
    xmr_actual = (xmr_df.iloc[start:,].values.ravel()).tolist()

    idx_actual = [i for i in range(0, len(xmr_actual))]
    json_actuals = []

    for actual,idx in zip(xmr_actual, idx_actual):

        json_actuals.append({'x': idx, 'y': actual })
    
    
    
    idx_pred = [i for i in range(len(xmr_actual), len(xmr_actual)+14)]
    json_preds = []
    json_preds.append(json_actuals[-1])
    

    close1 = xmr_pred.close_1.values
    close2 = xmr_pred.close_2.values
    close3 = xmr_pred.close_3.values
    close4 = xmr_pred.close_4.values
    close5 = xmr_pred.close_5.values
    close6 = xmr_pred.close_6.values
    close7 = xmr_pred.close_7.values
    close8 = xmr_pred.close_8.values
    close9 = xmr_pred.close_9.values
    close10 = xmr_pred.close_10.values
    close11 = xmr_pred.close_11.values
    close12 = xmr_pred.close_12.values
    close13 = xmr_pred.close_13.values
    close14 = xmr_pred.close_14.values

    preds = [close1[0],close2[0], close3[0], close4[0], close5[0], close6[0], close7[0], close8[0], close9[0], close10[0], close11[0], close12[0], close13[0], close14[0]]
    

    for pred,idx in zip(preds, idx_pred):

        json_preds.append({'x':idx ,'y':pred})

    
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_predictions(df):

    import numpy as np
    import ta
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
    import tensorflow as tf
    

    data_df = df
    data_df = data_df.drop(columns=['time','volumefrom'])

    data_df = data_df.dropna()

    high = data_df.high
    low = data_df.low
    close = data_df.close
    volume = data_df.volumeto

    start = 0
    end = len(data_df)


    x_axis = [i for i in range(start,len(data_df))]


    RSI = ta.momentum.StochRSIIndicator(close)
    rsi_k = RSI.stochrsi_k()
    rsi_d = RSI.stochrsi_d()

    MACD_indicator = ta.trend.MACD(close)
    macd = MACD_indicator.macd_diff()

    KD = ta.momentum.StochasticOscillator(close, high, low)
    kd = KD.stoch()

    OBV = ta.volume.OnBalanceVolumeIndicator(close, volume)
    obv = OBV.on_balance_volume()

    atr = ta.volatility.average_true_range(high, low, close)

    data_df = data_df.assign(rsi_k=rsi_k)
    data_df = data_df.assign(rsi_d=rsi_d)
    data_df = data_df.assign(macd=macd)
    data_df = data_df.assign(kd=kd)
    data_df = data_df.assign(obv=obv)
    data_df = data_df.assign(atr=atr)

    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    data_df = data_df.drop(columns=['volumeto'])
    
    x = len(data_df)%8
    n = len(data_df)-x


    temporal = [[0,1,0,1,0,1,0,1] for i in range(0,n,8)]

    temporal = np.asarray(temporal).ravel().tolist()

    for i in range(0,x):
        
        if i < 4:
            temporal.append(0)
        else:
            temporal.append(1)
   

    data_df = data_df.assign(time=temporal)


    #-------------------------------------------------------------------------------------------------------------------------
    #_________________________________________________________________________________________________________________________
    #_________ SPLIT DATASET - TRAIN/TEST ____________________________________________________________________________________



    in_window = 70 
    out_window = 14 
    test_len = 0 

    num_features = len(data_df.columns)

    dataset_len = len(data_df)
    train_len = len(data_df)-test_len

    #last_seen_close = close_df[train_len]


    data = data_df.iloc[:,0:num_features].values       

    #____________________________________________
    #---------- standard scaling ----------------

    sc = StandardScaler()
    sc2 = StandardScaler()

    train_data = sc.fit_transform(data[:train_len,:])
    train_close = sc2.fit_transform(np.asarray(close[:train_len]).reshape(-1,1))

    #plt.hist(train_data,bins=30)

    #____________________________________________
    #___________Train/Validation Data:___________

    x_train = []
    y_train = []


    x_valid = []
    y_valid = []

    for i in range(in_window,len(train_data)-out_window+1):
        
        x_train.append(train_data[i-in_window:i,:])
        y_train.append(train_close[i:i+out_window])
    
        if i < len(train_data)-out_window-out_window+1:
            x_valid.append(train_data[i-in_window+out_window:i+out_window,:])
            y_valid.append(train_close[i+out_window:i+out_window+out_window])

        

    #______reshape train-data to lstm expected shape_______

    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train).astype('float32')

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],num_features))

    print(x_train.shape,y_train.shape)

    x_valid = np.array(x_valid).astype('float32')
    y_valid = np.array(y_valid).astype('float32')

    x_valid = np.reshape(x_valid,(x_valid.shape[0],x_valid.shape[1],num_features))

    print(x_valid.shape,y_valid.shape)


    #______________Test Data:____________________

    #____________________________________________
    #--------- standard-scaling (test) ----------

    test_data = sc.transform(data[train_len-in_window:,:])

    #test_close = sc2.transform(np.asarray(close[train_len-in_window:]).reshape(-1,1))

    #____________________________________________
    #--------- log-transformed (test) -----------


    #____________________________________________

    x_test = []
    #y_test = []


    x_test.append(test_data[:,:])
    x_test = np.asarray(x_test).astype('float32')
    #y_test = np.asarray(y_test).astype('float32')

    x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],num_features))



    #_________________________________________________________________________________________________________________________

    def build_model(in_window, out_window, num_features):
        
        inputs = tf.keras.layers.Input(shape=(in_window, num_features))
        
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(inputs)
        
        layer =  tf.keras.layers.Dense(36, kernel_initializer='lecun_normal', activation='selu')(layer)

        layer =  tf.keras.layers.Dropout(0.4)(layer)

        outputs = tf.keras.layers.Dense(out_window)(layer)
        
        model =tf.keras.models.Model(inputs, outputs)
        
   
        #opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        opt = 'sgd'
        
        #loss = tf.keras.losses.Huber() 
        loss = 'mean_squared_error'
        
        model.compile(optimizer=opt, loss=loss, metrics=['mape'])
        
        return model
        
        
        
        
    model_dnn = build_model(in_window, out_window, num_features)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model_dnn.summary()
    hist_simple = model_dnn.fit(x_train, y_train, epochs=45, batch_size=8, callbacks=[callback], shuffle=False, validation_data=(x_valid, y_valid))


    y_pred = model_dnn.predict(x_test)


    #for y, pred in zip(y_test, y_pred):
    #    print("MSE: ", mean_squared_error(y, pred))
    #    print("MAPE: ", mean_absolute_percentage_error(y, pred))
    #    print('________________________________')
        
    y_pred = sc2.inverse_transform(y_pred)
    

    y_pred = y_pred[0]
    return y_pred
