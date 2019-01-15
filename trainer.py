
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import datetime
import sys
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import MinMaxScaler
# import util
import datetime


# In[6]:


class Trainer(object):
    def __init__(self,filepath1,filepath2,filepath3):
        self.filepath1 = filepath1
        self.filepath2 = filepath2
        self.filepath3 = filepath3
    def read_data(self):
        df_sr = pd.read_csv(self.filepath1)
        df_week = pd.read_csv(self.filepath2)
        df_lc = pd.read_csv(self.filepath3)
        df_sr = pd.merge(df_sr,df_week,on=['week_begin','week_end','week_of_year','week_from_cny',
                                     'dayOff'],how='inner')
        df_sr = pd.merge(df_sr,df_lc,on=['LC','product','sub_category','category','temp','off_on_ratio'],how='inner',suffixes='lr')
        return df_sr
        
    def preprocess(df):
        df.week_begin = pd.to_datetime(df.week_begin)
        df.week_end = pd.to_datetime(df.week_end)
        return df
    
    def split_train_test(df_all,input_sequence_length,pred_length,week_length=156,lag=1): 
    #### Input data formatting   
        date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df_all.week_begin[0:week_length]]),
                                  data=[i for i in range(len(df_all.week_begin[0:week_length]))])
        train_last_week_begin = date_to_index.index[0]
        test_last_week_begin = week_length - pred_length - input_sequence_length - lag

        val_last_week_end = week_length - pred_length - lag
        val_last_week_begin = val_last_week_end - pred_length - input_sequence_length -lag

        train_last_week_end = val_last_week_end - pred_length - lag

        train_time_split = date_to_index.index[train_last_week_end]
        test_time_split = date_to_index.index[test_last_week_begin]   
        val_time_split_begin = date_to_index.index[val_last_week_begin]   
        val_time_split_end = date_to_index.index[val_last_week_end] 

        train = df_all[df_all['week_begin']<=pd.to_datetime(train_time_split)]
        test = df_all[df_all['week_begin']>=pd.to_datetime(test_time_split)]
        val = df_all[(df_all['week_begin']>=pd.to_datetime(val_time_split_begin))&
                    (df_all['week_begin']<=pd.to_datetime(val_time_split_end))]

        return train,test,val

