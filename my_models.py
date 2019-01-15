
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate,Activation,BatchNormalization,SpatialDropout1D
from keras.optimizers import Adam,rmsprop


# In[5]:


class wavenet(object):
    ''' XGBoost决策树模型 '''
    def __init__(self,n_filters = 32,filter_width = 2,dilation_rates = [2**i for i in range(9)],
                 epochs = 200,batch_size = 128,val_batch_size = 64,steps_per_epoch = 500,name='wavenet'):
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.dilation_rates = dilation_rates
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.name = name
    def build_model(self,FEATURE_LENGTH,PRED_LENGTH):
        history_seq = Input(shape=(None, FEATURE_LENGTH))
        x = history_seq

        for dilation_rate in self.dilation_rates:
            x = Conv1D(filters=self.n_filters,
                       kernel_size=self.filter_width, 
                       padding='causal',
                       dilation_rate=dilation_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(.2)(x)
        x = Dense(1)(x)
        def slice(x, PRED_LENGTH):
            return x[:,-PRED_LENGTH:,:]
        
        pred_seq_train = Lambda(slice, arguments={'PRED_LENGTH':18})(x)
        model = Model(history_seq, pred_seq_train)
        return model


# In[9]:


from keras.layers import SpatialDropout1D
class waveadded(object):
    
        ################## CNN+dropout#####################
    '''
    seems to be easily overfitting--high val loss and probably skip-out
    1st solution: CNN+spatialdropout
    2st solution: LSTM wrapped--(attemp to learn better?)
    '''
    def __init__(self,n_filters = 32,filter_width = 2,dilation_rates = [2**i for i in range(9)],
                 epochs = 200,batch_size = 128,val_batch_size = 64,steps_per_epoch = 500,
                 dropout_rate = .2,name='waveadded'):
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.dilation_rates = dilation_rates
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.dropout_rate = dropout_rate
        self.name = name

    # define an input history series and pass it through a stack of dilated causal convolutions. 
    def build_model(self,FEATURE_LENGTH,PRED_LENGTH):
        
        history_seq = Input(shape=(None, FEATURE_LENGTH))
        x = history_seq
        name = ''

        for dilation_rate in self.dilation_rates:
            x = Conv1D(filters= self.n_filters,
                       kernel_size= self.filter_width, 
                       padding='causal',
                       dilation_rate=dilation_rate)(x)
            x = SpatialDropout1D(self.dropout_rate, name=name + 'spatial_dropout1d_%d_%f' % (dilation_rate*2,self.dropout_rate))(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(.2)(x)
        x = Dense(1)(x)

        # extract the last 18 time steps as the training target
        def slice(x, PRED_LENGTH):
            return x[:,-PRED_LENGTH:,:]

        pred_seq_train = Lambda(slice, arguments={'PRED_LENGTH':18})(x)

        model = Model(history_seq, pred_seq_train)
        return model


# In[39]:


############ wavenet+res_block##################
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate,SpatialDropout1D,add
from keras.optimizers import Adam
class tcn(object):
    def __init__(self,n_filters = 32,filter_width = 2,dilation_rates = [2**i for i in range(9)],
                 epochs = 200,batch_size = 128,val_batch_size = 64,steps_per_epoch = 500,
                 PRED_LENGTH=18,name='tcn'):
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.dilation_rates = dilation_rates
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.PRED_LENGTH = PRED_LENGTH
        self.name = name
        
        ############## Residual Network Added #################
    def residual_block(self,x, s, i, activation, nb_filters, kernel_size, padding, 
                       dropout_rate=.1,name=''):
        # type: (Layer, int, int, str, int, int, float, str) -> Tuple[Layer, Layer]
        original_x = x
        # print('input size-----',original_x.get_shape())
        x = BatchNormalization(axis=-1,name='bn_layer_%d_s%d_%f' % (i, s, dropout_rate))(x)
        x = Activation('linear')(x)
        conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=i, padding=padding)(x)
        x = Activation('linear')(x)    
        x = SpatialDropout1D(dropout_rate, name=name + 'spatial_dropout1d_%d_s%d_%f' % (i*2, s, dropout_rate))(x)
        # print('-----------------2dilate',x.get_shape())
        short_cut = Conv1D(nb_filters, 1, padding='same')(x)
        # 1x1 conv.
        x = Conv1D(nb_filters, 1, padding='same')(x)
        residual_x =add([original_x, x])
        return residual_x,short_cut
    
    def tcn_model(self,dilation_rate_range,nb_filters,inputs,padding,n_stacks,kernel_size,
                 dropout_rate,activation,
                 use_skip_connections=True,return_sequences=True,PRED_LENGTH=18):
        # define an input history series and pass it through a stack of dilated causal convolutions. 
    
        x = inputs
        x = Conv1D(filters=nb_filters, padding=padding,dilation_rate=1,kernel_size=self.filter_width)(x)
        S = []
        for s in range(n_stacks):
            for i in dilation_rate_range:
                x, short_cut = self.residual_block(x, s, i, activation, self.n_filters,kernel_size, padding, dropout_rate,activation)
                # print(x.get_shape())
                S.append(short_cut)
        if use_skip_connections:
            skip_connection = add(S)
            x = skip_connection

        x = Dense(128, activation='relu')(x)
        x = Dropout(.1)(x)
        x = Dense(1)(x)

        if not return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        else:
            x = Lambda(lambda tt: tt[:, -PRED_LENGTH:, :])(x)
        return x
        
    def build_model(self,FEATURE_LENGTH):
        history_seq = Input(shape=(None, FEATURE_LENGTH))
        xx = history_seq

        pred_seq_target = self.tcn_model(self.dilation_rates,self.n_filters,xx,'causal',1,self.filter_width,.1,'relu')
        model = Model(history_seq,pred_seq_target)
        return model

