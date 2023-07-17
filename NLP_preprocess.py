# import spacy
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn import svm
# import re
import numpy as np
import pandas as pd
import re
import io
import json
from konlpy.tag import Okt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
DATA_PATH = 'DATA'
train_data = pd.read_csv(DATA_PATH+'/ratings_train.txt', header = 0, delimiter='\t', quoting=3)

class Category:
    positve = '긍정'
    negative = '부정'

stop_words = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한', '에']
pattern = '[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]' 
clean_train_review = []
okt = Okt()
def preprocessing(review, okt, remove_stopwords = False, stop_words = []):
    text = re.sub(pattern=pattern, repl=' ', string=review) #한글 제외 언어들을 삭제
    word_text = okt.morphs(text, stem = True) #Okt 객체를 활용하여 형태소 단어로 나눔
    
    if (remove_stopwords == True):
        word_text = [token for token in word_text if not token in stop_words]
    return word_text
for review in train_data['document']:
    if(type(review) == str):
        clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words = stop_words))
    else:
        clean_train_review.append([])

test_data = pd.read_csv(DATA_PATH+'/ratings_test.txt', header = 0, delimiter='\t', quoting=3)
clean_test_review = []
for review in test_data['document']:
  if type(review) == str:
    clean_test_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
  else:
    clean_test_review.append([])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_review)
train_sequences = tokenizer.texts_to_sequences(clean_train_review)
test_sequences = tokenizer.texts_to_sequences(clean_test_review)

word_vocab = tokenizer.word_index #단어 사전 형태
MAX_SEQUENCE_LENGTH = 8 #문장 최대 길이

train_inputs = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')

train_labels = np.array(train_data['label'])

test_inputs = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')

test_labels =  np.array(test_data['label'])

DEFAULT_DATA_PATH_SAVE = 'DATA/'
DATA_PATH_SAVE = 'CLEAN_DATA/'
TRAIN_INPUT_DATA = 'nsmc_train_input.npy'
TRAIN_LABEL_DATA = 'nsmc_train_label.npy'
TEST_INPUT_DATA = 'nsmc_test_input.npy'
TEST_LABEL_DATA = 'nsmc_test_label.npy'
DATA_CONFIGS = 'data_configs.json'

data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) + 1

if not os.path.exists(DEFAULT_DATA_PATH_SAVE + DATA_PATH_SAVE):
   os.makedirs(DEFAULT_DATA_PATH_SAVE + DATA_PATH_SAVE)

#전처리 학습데이터 넘파이로 저장
np.save(open(DEFAULT_DATA_PATH_SAVE+DATA_PATH_SAVE+TRAIN_INPUT_DATA,'wb'),train_inputs)
np.save(open(DEFAULT_DATA_PATH_SAVE+DATA_PATH_SAVE+TRAIN_LABEL_DATA,'wb'),train_labels)
#전처리 테스트데이터 넘파이로 저장
np.save(open(DEFAULT_DATA_PATH_SAVE+DATA_PATH_SAVE+TEST_INPUT_DATA,'wb'),test_inputs)
np.save(open(DEFAULT_DATA_PATH_SAVE+DATA_PATH_SAVE+TEST_LABEL_DATA,'wb'),test_labels)

#데이터 사전 json으로 저장
json.dump(data_configs,open(DEFAULT_DATA_PATH_SAVE + DATA_PATH_SAVE + DATA_CONFIGS,'w'),ensure_ascii=False)
import pickle
with open(DEFAULT_DATA_PATH_SAVE + DATA_PATH_SAVE + 'tokenizer.pickle', 'wb') as handle:
    pickle.dump(word_vocab, handle)
