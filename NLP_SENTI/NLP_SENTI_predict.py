import re
import json
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import keras

okt = Okt()
tokenizer  = Tokenizer()

DATA_CONFIGS = 'data_configs_test.json'
prepro_configs = json.load(open('DATA/CLEAN_DATA/'+DATA_CONFIGS,'r'))

with open('DATA/CLEAN_DATA/tokenizer_senti.pickle','rb') as handle:
    word_vocab = pickle.load(handle)

prepro_configs['vocab'] = word_vocab

tokenizer.fit_on_texts(word_vocab)

MAX_LENGTH = 8
model = keras.models.load_model('DATA/my_senti_models')
model.load_weights('DATA/DATA_SENTI_OUT/cnn_classifier_kr/weights.h5')

while True:
    sentence = input('감성분석할 문장을 입력해 주세요.: ')
    if sentence == '끝':
        break
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]','', sentence)
    stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한']
    sentence = okt.morphs(sentence, stem=True)
    sentence = [word for word in sentence if not word in stopwords]
    vector = tokenizer.texts_to_sequences([sentence])
    pad_new = pad_sequences(vector, maxlen=MAX_LENGTH)

    predictions = model.predict(pad_new)
    predictions = float(predictions[0])

    if predictions > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))


# import re
# import json
# from konlpy.tag import Okt
# from keras.utils import pad_sequences
# from keras.preprocessing.text import Tokenizer
# import pickle
# import keras

# okt = Okt()
# tokenizer  = Tokenizer()

# DATA_CONFIGS = 'data_configs_test.json'
# prepro_configs = json.load(open('DATA/CLEAN_DATA/'+DATA_CONFIGS,'r')) #TODO 데이터 경로 설정

# #TODO 데이터 경로 설정
# with open('DATA/CLEAN_DATA/tokenizer_senti.pickle','rb') as handle:
#     word_vocab = pickle.load(handle)

# prepro_configs['vocab'] = word_vocab

# tokenizer.fit_on_texts(word_vocab)

# MAX_LENGTH = 8 #문장최대길이
# model = keras.models.load_model('DATA/my_senti_models/') #TODO 데이터 경로 설정
# model.load_weights('DATA/DATA_SENTI_OUT/cnn_classifier_kr/weights.h5') #TODO 데이터 경로 설정
# while True:
#     sentence = input('감성분석할 문장을 입력해 주세요.: ')
#     if sentence == '끝':
#         break
#     sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]','', sentence)
#     stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한'] # 불용어 추가할 것이 있으면 이곳에 추가
#     sentence = okt.morphs(sentence, stem=True) # 토큰화
#     sentence = [word for word in sentence if not word in stopwords] # 불용어 제거
#     vector  = tokenizer.texts_to_sequences(sentence)
#     pad_new = pad_sequences(vector, maxlen = MAX_LENGTH) # 패딩

#     #학습한 모델 불러오기
  
#     predictions = model.predict(pad_new)
#     predictions = float(predictions.squeeze(-1)[0])

#     if(predictions > 0.5):
#         print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
#     else:
#         print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))
    