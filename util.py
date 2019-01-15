
# coding: utf-8

# In[1]:


from __future__ import absolute_import
import numpy as np
import pandas as pd
import trainer


# In[2]:


def get_feature_num(df_sample):
    #std_feature = StandardScaler.fit_transform(feature_mtx)
    enc_mtx = enc.transform(df_sample).toarray()
    return (enc_mtx.shape[1]+len(SR_FEATURE))


# In[3]:


def get_ts_info(df):
    df_group = df.groupby(['LC','product'])
    feature_data = []
    sr_data = []
    for i in df_group:
        # print(s_data.shape)
        s_sample = i[1].sort_values(by=['week_begin'],ascending=True)
        row = i[1].shape[0]
        for j in range(0,row-whole_ts,1):
            x_col = s_sample.iloc[j:(j+whole_ts)][FEATURE]
            x = x_col.iloc[:,1:]
            y = x_col.iloc[:,0]
            feature_data.append(x.values)
            sr_data.append(y.values)
    return feature_data,sr_data


# In[4]:


def get_feature_mtx(x,f_start):
    #print(x[:,f_start:].shape)
    mtx = enc.transform(x[:,f_start:]).toarray()
    #print(mtx.shape)
    data = np.concatenate([x[:,:f_start],mtx],axis=1)
    return data


# In[5]:


def get_ts_matrix(df,input_sequence_length,target_sequence_length):
    ############## get time series data #################
    whole = input_sequence_length+target_sequence_length
    TS_mean = []
    X = df[:whole,0]
    X = np.log1p(X)
    ts_mean = X.mean()
    TS_mean.append(ts_mean)
    X = X/ts_mean
    Y = df[:input_sequence_length,0]
    
    feature = df[:whole,1:]
    # other_features = df[SR_FEATURE][:whole]
    # feature = np.hstack([stable_features,other_features])
    #Y_feature = np.hstack([df[ENC_COL][input_sequence_length:],df[SR_FEATURE][input_sequence_length:]])
    return X,feature,Y,TS_mean

