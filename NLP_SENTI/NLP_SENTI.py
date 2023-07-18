import numpy as np
import pandas as pd
import re
import json
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

DATA_PATH = 'DATA/'
pos_data = pd.read_csv(DATA_PATH + 'pos_pol_word.txt', delimiter='/t', header=None)
train_data = pd.read_csv(DATA_PATH+'ratings_train.txt', header = 0, delimiter='\t', quoting=3)
def preprocessing(review, okt, remove_stopwords = False, stop_words =[]):
  #함수인자설명
  # review: 전처리할 텍스트
  # okt: okt객체를 반복적으로 생성하지 않고 미리 생성 후 인자로 받음
  # remove_stopword: 불용어를 제거할지 여부 선택. 기본값 False
  # stop_words: 불용어 사전은 사용자가 직접 입력, 기본값 빈 리스트

  # 1. 한글 및 공백 제외한 문자 모두 제거
  review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]','',review)
  
  #2. okt 객체를 활용하여 형태소 단어로 나눔
  word_review = okt.morphs(review_text,stem=True)

  if remove_stopwords:
    #3. 불용어 제거(선택)
    word_review = [token for token in word_review if not token in stop_words]
  return word_review

stop_words = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한']
okt = Okt()
clean_pos_review = []
clean_train_review = []
clean_neg_review = []
for review in train_data['document']:
    if(type(review) == str):
        clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words = stop_words))
    else:
        clean_train_review.append([])
for review in pos_data[0]:
  # 리뷰가 문자열인 경우만 전처리 진행
  if type(review) == str:
    clean_pos_review.append(preprocessing(review,okt,remove_stopwords=True,stop_words= stop_words))
  else:
    clean_pos_review.append([]) #str이 아닌 행은 빈칸으로 놔두기

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_pos_review)
tokenizer.fit_on_texts(clean_train_review)
train_sequences = tokenizer.texts_to_sequences(clean_train_review)
pos_sequences = tokenizer.texts_to_sequences(clean_pos_review)


MAX_SEQUENCE_LENGTH = 8 #문장 최대 길이

#학습 데이터
pos_inputs = pad_sequences(pos_sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post')

#학습 데이터 라벨 벡터화
pos_labels = np.full((len(pos_inputs),), 1)

neg_data = pd.read_csv(DATA_PATH + 'neg_pol_word.txt', delimiter='/t', header=None)
for review in neg_data[0]:
  # 리뷰가 문자열인 경우만 전처리 진행
  if type(review) == str:
    clean_neg_review.append(preprocessing(review,okt,remove_stopwords=True,stop_words= stop_words))
  else:
    clean_neg_review.append([]) #str이 아닌 행은 빈칸으로 놔두기
tokenizer.fit_on_texts(clean_neg_review)
neg_sequences = tokenizer.texts_to_sequences(clean_neg_review)
train_inputs = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')

train_labels = np.array(train_data['label'])
neg_inputs = pad_sequences(neg_sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post')
word_vocab = tokenizer.word_index #단어사전형태
#학습 데이터 라벨 벡터화
neg_labels = np.full((len(neg_inputs),), 0)

DEFAULT_PATH  = 'DATA/' # 경로지정
DATA_PATH = 'CLEAN_DATA/' #.npy파일 저장 경로지정
POS_INPUT_DATA = 'nsmc_pos_input.npy'
POS_LABEL_DATA = 'nsmc_pos_label.npy'
NEG_INPUT_DATA = 'nsmc_neg_input.npy'
NEG_LABEL_DATA = 'nsmc_neg_label.npy'
DATA_CONFIGS = 'data_configs_test.json'
DEFAULT_DATA_PATH_SAVE = 'DATA/'
DATA_PATH_SAVE = 'CLEAN_DATA/'
TRAIN_INPUT_DATA = 'nsmc_train_input.npy'
TRAIN_LABEL_DATA = 'nsmc_train_label.npy'

data_configs={}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) + 1

#전처리한 데이터들 파일로저장
import os

if not os.path.exists(DEFAULT_PATH + DATA_PATH):
  os.makedirs(DEFAULT_PATH+DATA_PATH)

#전처리 학습데이터 넘파이로 저장
np.save(open(DEFAULT_PATH+DATA_PATH+POS_INPUT_DATA,'wb'),pos_inputs)
np.save(open(DEFAULT_PATH+DATA_PATH+POS_LABEL_DATA,'wb'),pos_labels)
np.save(open(DEFAULT_DATA_PATH_SAVE+DATA_PATH_SAVE+TRAIN_INPUT_DATA,'wb'),train_inputs)
np.save(open(DEFAULT_DATA_PATH_SAVE+DATA_PATH_SAVE+TRAIN_LABEL_DATA,'wb'),train_labels)
#전처리 테스트데이터 넘파이로 저장
np.save(open(DEFAULT_PATH+DATA_PATH+NEG_INPUT_DATA,'wb'),neg_inputs)
np.save(open(DEFAULT_PATH+DATA_PATH+NEG_LABEL_DATA,'wb'),neg_labels)

#데이터 사전 json으로 저장
json.dump(data_configs,open(DEFAULT_PATH + DATA_PATH + DATA_CONFIGS,'w'),ensure_ascii=False)
import pickle
with open(DEFAULT_PATH + DATA_PATH + 'tokenizer_senti.pickle', 'wb') as handle:
    pickle.dump(word_vocab, handle)
