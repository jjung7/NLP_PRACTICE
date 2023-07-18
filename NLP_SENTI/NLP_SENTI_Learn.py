import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import save_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

# 전처리 데이터 불러오기
DATA_PATH = 'DATA/CLEAN_DATA/'
DATA_SENTI_OUT = 'DATA/DATA_SENTI_OUT/'
INPUT_POS_DATA = 'nsmc_pos_input.npy'
LABEL_POS_DATA = 'nsmc_pos_label.npy'
INPUT_NEG_DATA = 'nsmc_neg_input.npy'
LABEL_NEG_DATA = 'nsmc_neg_label.npy'
INPUT_TRAIN_DATA = 'nsmc_train_input.npy'
LABEL_TRAIN_DATA = 'nsmc_train_label.npy'
DATA_CONFIGS = 'data_configs_test.json'
train_input = np.load(open(DATA_PATH + INPUT_TRAIN_DATA,'rb'))
train_input = pad_sequences(train_input,maxlen=train_input.shape[1])
train_label = np.load(open(DATA_PATH + LABEL_TRAIN_DATA,'rb'))
pos_input = np.load(open(DATA_PATH + INPUT_POS_DATA, 'rb'))
pos_input = pad_sequences(pos_input, maxlen=pos_input.shape[1])
pos_label = np.load(open(DATA_PATH + LABEL_POS_DATA, 'rb'))
neg_input = np.load(open(DATA_PATH + INPUT_NEG_DATA, 'rb'))
neg_input = pad_sequences(neg_input, maxlen=neg_input.shape[1])
neg_label = np.load(open(DATA_PATH + LABEL_NEG_DATA, 'rb'))
prepro_configs = json.load(open(DATA_PATH + DATA_CONFIGS, 'r'))

model_name = 'cnn_classifier_kr'
BATCH_SIZE = 512
NUM_EPOCHS = 10
VALID_SPLIT = 0.1
MAX_LEN_POS = pos_input.shape[1]
MAX_LEN_NEG = neg_input.shape[1]

input_data = np.concatenate((pos_input, neg_input,train_input), axis=0)
label_data = np.concatenate((pos_label, neg_label,train_label), axis=0)

data = list(zip(input_data, label_data))

# Shuffle the data
np.random.shuffle(data)

# Separate input_data and label_data
shuffled_input_data, shuffled_label_data = zip(*data)

# Convert the shuffled_input_data and shuffled_label_data back to numpy arrays
shuffled_input_data = np.array(shuffled_input_data)
shuffled_label_data = np.array(shuffled_label_data)
kargs = {'model_name': model_name, 'vocab_size': prepro_configs['vocab_size'], 'embedding_size': 128,
         'num_filters': 100, 'dropout_rate': 0.5, 'hidden_dimension': 250, 'output_dimension': 1}


class CNNClassifier(tf.keras.Model):

    def __init__(self, **kargs):
        super(CNNClassifier, self).__init__(name=kargs['model_name'])
        self.embedding = layers.Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embedding_size'])
        self.conv_list = [
            layers.Conv1D(filters=kargs['num_filters'], kernel_size=kernel_size, padding='valid',
                          activation=tf.keras.activations.relu,
                          kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3)) for kernel_size in [3, 4, 5]]
        self.pooling = layers.GlobalMaxPooling1D()
        self.dropout = layers.Dropout(kargs['dropout_rate'])
        self.fc1 = layers.Dense(units=kargs['hidden_dimension'],
                                activation=tf.keras.activations.relu,
                                kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
        self.fc2 = layers.Dense(units=kargs['output_dimension'],
                                activation=tf.keras.activations.sigmoid,
                                kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x = tf.concat([self.pooling(conv(x)) for conv in self.conv_list], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = CNNClassifier(**kargs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

# 검증 정확도를 통한 EarlyStopping 기능 및 모델 저장 방식 지정
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)
checkpoint_path = DATA_SENTI_OUT + model_name + '\weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))

cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True
)

history = model.fit(shuffled_input_data, shuffled_label_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

# 모델 저장하기
save_model(model, 'DATA/my_senti_models')



# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras import layers
# from tensorflow.keras.models import save_model
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import json
# from tqdm import tqdm

# #전처리 데이터 불러오기
# DATA_PATH = 'DATA/CLEAN_DATA/'
# DATA_SENTI_OUT = 'DATA/DATA_SENTI_OUT/'
# INPUT_POS_DATA = 'nsmc_pos_input.npy'
# LABEL_POS_DATA = 'nsmc_pos_label.npy'
# INPUT_NEG_DATA = 'nsmc_neg_input.npy'
# LABEL_NEG_DATA = 'nsmc_neg_label.npy'
# DATA_CONFIGS = 'data_configs_test.json'

# pos_input = np.load(open(DATA_PATH + INPUT_POS_DATA,'rb'))
# pos_input = pad_sequences(pos_input,maxlen=pos_input.shape[1])
# pos_label = np.load(open(DATA_PATH + LABEL_POS_DATA,'rb'))
# neg_input = np.load(open(DATA_PATH + INPUT_NEG_DATA,'rb'))
# neg_input = pad_sequences(neg_input,maxlen=neg_input.shape[1])
# neg_label = np.load(open(DATA_PATH + LABEL_NEG_DATA,'rb'))
# prepro_configs = json.load(open(DATA_PATH+DATA_CONFIGS,'r'))

# model_name= 'cnn_classifier_kr'
# BATCH_SIZE = 512
# NUM_EPOCHS = 10
# VALID_SPLIT = 0.1
# MAX_LEN_POS = pos_input.shape[1]
# MAX_LEN_NEG = neg_input.shape[1]

# input_data = np.concatenate((pos_input, neg_input), axis=0)
# label_data = np.concatenate((pos_label, neg_label), axis=0)
# kargs={'model_name': model_name, 'vocab_size':prepro_configs['vocab_size'],'embbeding_size':128, 'num_filters':100,'dropout_rate':0.5, 'hidden_dimension':250,'output_dimension':1}
# class CNNClassifier(tf.keras.Model):

#     def __init__(self, **kargs):
#         super(CNNClassifier, self).__init__(name=kargs['model_name'])
#         self.embedding = layers.Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embbeding_size'])
#         self.conv_list = [layers.Conv1D(filters=kargs['num_filters'], kernel_size=kernel_size, padding='valid',activation = tf.keras.activations.relu,
#                                         kernel_constraint = tf.keras.constraints.MaxNorm(max_value=3)) for kernel_size in [3,4,5]]
#         self.pooling = layers.GlobalMaxPooling1D()
#         self.dropout = layers.Dropout(kargs['dropout_rate'])
#         self.fc1 = layers.Dense(units=kargs['hidden_dimension'],
#                                 activation = tf.keras.activations.relu,
#                                 kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
#         self.fc2 = layers.Dense(units=kargs['output_dimension'],
#                                 activation=tf.keras.activations.sigmoid,
#                                 kernel_constraint= tf.keras.constraints.MaxNorm(max_value=3.))
    
#     def call(self, inputs):
#       x = self.embedding(inputs)
#       x = self.dropout(x)
#       x = tf.concat([self.pooling(conv(x)) for conv in self.conv_list], axis=1)
#       x = self.fc1(x)
#       x = self.fc2(x)
#       return x
# # print("pos_input shape:", pos_input.shape)
# # print("neg_input shape:", neg_input.shape)
# # print("pos_label shape:", pos_label.shape)
# # print("neg_label shape:", neg_label.shape)

# # indices = np.arange(input_data.shape[0])
# # np.random.shuffle(indices)
# # input_data = input_data[indices]
# # label_data = label_data[indices]
# model = CNNClassifier(**kargs)
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss = tf.keras.losses.BinaryCrossentropy(),
#               metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')])

# #검증 정확도를 통한 EarlyStopping 기능 및 모델 저장 방식 지정
# earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)
# checkpoint_path = DATA_SENTI_OUT + model_name +'\weights.h5'
# checkpoint_dir = os.path.dirname(checkpoint_path)

# if os.path.exists(checkpoint_dir):
#   print("{} -- Folder already exists \n".format(checkpoint_dir))
# else:
#   os.makedirs(checkpoint_dir, exist_ok=True)
#   print("{} -- Folder create complete \n".format(checkpoint_dir))

# cp_callback = ModelCheckpoint(
#     checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only = True,
#     save_weights_only=True
# )
# # print(pos_input.shape)
# # print(neg_input.shape)
# # print(pos_label.shape)
# # print(neg_label.shape)

# history = model.fit(input_data, label_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
#                     validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])
# # 모델 저장하기
# save_model(model, 'DATA/my_senti_models')