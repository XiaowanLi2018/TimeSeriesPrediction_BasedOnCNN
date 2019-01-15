
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES = 1')
import trainer
import pandas as pd
import numpy as np
import util
import my_models
from util import *
from my_models import *


# In[2]:


# choose your model to predict
model_list = [wavenet,waveadded,tcn]
model_ob = model_list[0]()
m = model_ob.build_model(68,18)


# In[3]:


ENC_COL = ['week_from_cny',
 'off_on_ratio','week_of_year',
 'holiday_effect_noncny',
 'is_cny']
SR_FEATURE = [ 'sr','same_product_event_sr',
 'same_subcategory_event_sr',
 'same_category_event_sr',]
FEATURE = SR_FEATURE+ENC_COL
DATA = trainer.Trainer('../../TEST/synthetic_rnn/data_level4.csv','../../TEST/synthetic_rnn/dim_week.csv',
                      '../../TEST/synthetic_rnn/dim_product_lc.csv')
df_sr4 = DATA.read_data()
df_all = DATA.preprocess(df_sr4)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
enc = OneHotEncoder()
f_mtx = df_sr4[FEATURE].iloc[:,1:]
f_in = np.array(f_mtx)[:,(len(SR_FEATURE)-1):]
t_enc = enc.fit(f_in)

ENC_LENGTH = 18*4
PRED_LENGTH = 18
#fea_num = len(FEATURE_COL)
BATCH_SIZE = 128
whole_ts = PRED_LENGTH+ENC_LENGTH
TRAIN,TEST,VAL = DATA.split_train_test(df_all,ENC_LENGTH,PRED_LENGTH)
TEST_FEATURE_DATA,TEST_SR_DATA = get_ts_info(TEST,whole_ts,FEATURE)
FEATURE_LENGTH = 68


# In[4]:


steps_per_epoch = 500


# In[5]:


generator = data_gen()
test_gen = generator.data_generator(TEST_FEATURE_DATA[-1000:],TEST_SR_DATA[-1000:],batch_size=1000,
                                    feature_num = FEATURE_LENGTH,
                  steps_per_epoch=steps_per_epoch,
                           input_sequence_length=ENC_LENGTH,
                                    sr_num = len(SR_FEATURE),
                           target_sequence_length=PRED_LENGTH,
                            seed=30)


# In[41]:


from keras import models
bacth_size = 128

class predict(object):
    def __init__(self,filepath='../base02.h5',test_generator=test_gen,batch_size=128):
        self.filepath = filepath
        self.test_generator= test_gen
        self.batch_size = batch_size
        
    def predict_sequence(self,my_model,datagen,sample_size,pred_steps=18):
        pred_seq = np.zeros((sample_size,pred_steps,1))
        iterator = iter(datagen)
        input_sequence = next(iterator)
        target = input_sequence[1]
        x = input_sequence[0]
        for j in range(pred_steps):
            last_pred_step = my_model.predict(x)[:,-1,0]
            #print(last_pred_step.shape)
            pred_seq[:,j,0] = last_pred_step
            #print(input_sequence[0].shape)
            last_step = np.zeros((sample_size,1,FEATURE_LENGTH))
            last_step[:,0,:] = input_sequence[0][:,ENC_LENGTH+j,:]
            last_step[:,0,0] = last_pred_step
            input_ts = np.concatenate([input_sequence[0],last_step],axis=1)
            #print(input_ts.shape)
            x = input_ts
        return pred_seq,target
    
    def get_results(self):
        model_used = models.load_model(self.filepath)
        generator = data_gen()
        test_gen = generator.data_generator(TEST_FEATURE_DATA[-1000:],TEST_SR_DATA[-1000:],batch_size=1,
                                            feature_num = FEATURE_LENGTH,
                          steps_per_epoch=steps_per_epoch,
                                   input_sequence_length=ENC_LENGTH,
                                            sr_num=4,
                                   target_sequence_length=PRED_LENGTH,
                                    seed=30)
        predict,target = self.predict_sequence(model_used,test_gen,sample_size=1)
        return predict,target
    
    def predict_and_plot(self,feature_list, sr_list, sample_ind, my_model,sr_num,enc_tail_len=72,pred_steps=18):
        input_series = self.get_batch_input(feature_list, sr_list, sample_ind, enc_tail_len, pred_steps,sr_num)
        pred_series,target = self.get_results()
        # print(pred_series)

        #input_series = input_series.reshape(-1,1)
        pred_series = pred_series.reshape(-1,1)   
        mean = np.mean(np.log1p(sr_list[sample_ind]))
        sr = np.log1p(sr_list[sample_ind])/mean
        target_series = sr[enc_tail_len:].reshape(-1,1)
        '''
        mean = np.mean(np.log1p(sr_list[sample_ind]))
        pred = np.exp(mean*pred_series)
        '''
        encode_series_tail = sr[:72].reshape(-1,1)
        # print(encode_series_tail.shape)
        x_encode = enc_tail_len

        plt.figure(figsize=(10,6))   

        plt.plot(range(1,x_encode+1),encode_series_tail)
        plt.plot(range(x_encode,x_encode+pred_steps),target_series,color='orange')
        plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')

        plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
        plt.legend(['Encoding Series','Target Series','Predictions'])
    
    def get_batch_input(self,feature_list, sr_list, sample_ind, input_seq_length, pred_length,sr_num):
        s = np.zeros((1,whole_ts,FEATURE_LENGTH))
        x,feature,y,ts_mean_list = get_batch_matrix(feature_list,sr_list,sample_ind,ENC_LENGTH,PRED_LENGTH,sr_num)
        for i in range(whole_ts):
            s[0,i,:] = feature[i]
        s[0,:,0] = x
        enc_input = s[:,:ENC_LENGTH,:]
        dec_output = np.expand_dims(s[:,ENC_LENGTH:,0],axis=2)
        dec_input = get_teaching_force(FEATURE_LENGTH,enc_input,s[:,ENC_LENGTH:,:])
        in_data = np.concatenate([enc_input, dec_input], axis=1)
        return in_data


# In[42]:


p = predict()


# In[44]:


p.predict_and_plot(TEST_FEATURE_DATA, TEST_SR_DATA,120,m,4, enc_tail_len=72,pred_steps=18)

