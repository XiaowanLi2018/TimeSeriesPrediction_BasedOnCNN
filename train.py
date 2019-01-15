
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


ENC_COL = ['week_from_cny',
 'off_on_ratio','week_of_year',
 'holiday_effect_noncny',
 'is_cny']
SR_FEATURE = [ 'sr','same_product_event_sr',
 'same_subcategory_event_sr',
 'same_category_event_sr',]
FEATURE = SR_FEATURE+ENC_COL


# In[3]:


DATA = trainer.Trainer('../../TEST/synthetic_rnn/data_level4.csv','../../TEST/synthetic_rnn/dim_week.csv',
                      '../../TEST/synthetic_rnn/dim_product_lc.csv')
df_sr4 = DATA.read_data()
df_all = DATA.preprocess(df_sr4)


# In[4]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
enc = OneHotEncoder()
f_mtx = df_sr4[FEATURE].iloc[:,1:]
f_in = np.array(f_mtx)[:,(len(SR_FEATURE)-1):]
t_enc = enc.fit(f_in)


# In[5]:


df_sr4.head(5)


# In[6]:


ENC_LENGTH = 18*4
PRED_LENGTH = 18
#fea_num = len(FEATURE_COL)
BATCH_SIZE = 128
whole_ts = PRED_LENGTH+ENC_LENGTH
TRAIN,TEST,VAL = DATA.split_train_test(df_all,ENC_LENGTH,PRED_LENGTH)


# In[7]:


TRAIN_FEATURE_DATA,TRAIN_SR_DATA = get_ts_info(TRAIN,whole_ts,FEATURE)
VAL_FEATURE_DATA,VAL_SR_DATA = get_ts_info(VAL,whole_ts,FEATURE)


# In[8]:


def get_feature_num(df_sample):
    #std_feature = StandardScaler.fit_transform(feature_mtx)
    enc_mtx = enc.transform(df_sample).toarray()
    return (enc_mtx.shape[1]+len(SR_FEATURE))


# In[9]:


ss = TRAIN_FEATURE_DATA[10001][:,3:]
FEATURE_LENGTH = get_feature_num(ss)-1


# In[10]:


def get_feature_mtx(x,f_start):
    #print(x[:,f_start:].shape)
    mtx = enc.transform(x[:,f_start:]).toarray()
    #print(mtx.shape)
    data = np.concatenate([x[:,:f_start],mtx],axis=1)
    return data


# In[11]:


def get_batch_matrix(feature_list,sr_list,index,input_sequence_length,target_sequence_length):
    ############## get time series data #################
    whole = input_sequence_length+target_sequence_length
    TS_mean = []
    X = sr_list[index]
    X = np.log1p(X)
    ts_mean = X.mean()
    TS_mean.append(ts_mean)
    X = X/ts_mean
    #X_input = X[:input_sequence_length]
    Y = X[input_sequence_length:]
    
    feature = get_feature_mtx(feature_list[index],len(SR_FEATURE)-1)
    # f = get_enc_data(feature)
    # other_features = df[SR_FEATURE][:whole]
    # feature = np.hstack([stable_features,other_features])
    #Y_feature = np.hstack([df[ENC_COL][input_sequence_length:],df[SR_FEATURE][input_sequence_length:]])
    return X,feature,Y,TS_mean


# In[14]:


# choose your model to run
model_list = [wavenet,waveadded,tcn]
model_ob = model_list[0]()
model = model_ob.build_model(FEATURE_LENGTH,PRED_LENGTH)


# In[36]:


# train model with lagging_force
from keras import optimizers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
import keras.backend as kb
#import CLR
n_filters = 32 
filter_width = 2
dilation_rates = [2**i for i in range(9)] 
epochs = 200
batch_size = 128
val_batch_size = 64
steps_per_epoch = 500

earlystopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=1, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, 
    verbose=1, mode='auto', epsilon=None)
'''
# drop-cyclic lr schedule - DIY trainning period
def step_decay(epochs):
    initial_lr = 0.0005 
    period = 15
    drop = 0.5
    lr = initial_lr*math.pow(drop, math.floor((1+epochs)/period))
lr_scheduler = LearningRateScheduler(step_decay, verbose=1)
lr_scheduler = CLR.CyclicLR(base_lr=0.0005, max_lr=0.002,
                        step_size=200., mode='triangular2')
'''
callbacks = [earlystopping,lr_reducer]

rmsprop = optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=1e-9)
model.compile(rmsprop, loss='mean_absolute_error')

generator = data_gen()
train_data_generator = generator.data_generator(TRAIN_FEATURE_DATA,TRAIN_SR_DATA,batch_size=batch_size,
                                   steps_per_epoch=steps_per_epoch,
                                      feature_num = FEATURE_LENGTH,
                                   input_sequence_length=ENC_LENGTH,
                                   target_sequence_length=PRED_LENGTH,
                                    seed=100)
val_data_generator = generator.data_generator(VAL_FEATURE_DATA,VAL_SR_DATA,batch_size=val_batch_size,
                                   steps_per_epoch=steps_per_epoch,
                                      feature_num = FEATURE_LENGTH,
                                   input_sequence_length=ENC_LENGTH,
                                   target_sequence_length=PRED_LENGTH,
                                    seed=43)
#kb.clear_session()
history = model.fit_generator(train_data_generator, 
                    steps_per_epoch = steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_data_generator,
                             validation_steps=50)


# In[16]:


def save_model(save_model,fp):
    save_model.save(fp)
    save_model.save_weights(fp)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train','Valid'])


# In[39]:


TEST_FEATURE_DATA,TEST_SR_DATA = get_ts_info(TEST,whole_ts,FEATURE)
generator = data_gen()
test_gen = generator.data_generator(TEST_FEATURE_DATA,TEST_SR_DATA,batch_size=val_batch_size,
                                   steps_per_epoch=steps_per_epoch,
                                      feature_num = FEATURE_LENGTH,
                                   input_sequence_length=ENC_LENGTH,
                                   target_sequence_length=PRED_LENGTH,
                                    seed=100)


# In[23]:


from keras import models
dd = models.load_model('../base02.h5')


# In[37]:


dd.evaluate_generator(test_gen,steps=2000,verbose=1)

