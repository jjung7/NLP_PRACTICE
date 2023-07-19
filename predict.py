from __future__ import division
import re
import json
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import os
import keras
from test_mic import MicrophoneStream
import sys
import os
from google.cloud import speech
import pyaudio
from six.moves import queue
import datetime
# def init():
okt = Okt()
tokenizer = Tokenizer()
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
prepro_configs = json.load(open('DATA/CLEAN_DATA/data_configs.json','r'))
with open('DATA/CLEAN_DATA/tokenizer.pickle', 'rb') as handle:
    word_vocab = pickle.load(handle)

prepro_configs['vocab'] = word_vocab
tokenizer.fit_on_texts(word_vocab)
model = keras.models.load_model('DATA/my_models/') #TODO 데이터 경로 설정
model.load_weights('DATA/DATA_OUT/cnn_classifier_kr/weights.h5')
MAX_LENGTH = 8                                         
stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한'] # 불용어 추가할 것이 있으면 이곳에 추가

def listen_print_loop(responses):

    # num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        # if not result.alternatives:
        #     continue

        transcript = result.alternatives[0].transcript
        
        # overwrite_chars = " " * (num_chars_printed - len(transcript))

        if result.is_final:
           
            sentence = reversed(transcript)
            print(reversed(transcript))
            # print('전사:\n ' , transcript)
            # print('전사:\n ' , transcript , '응답 : \n' , response,  '결과 :\n' , result)
            # sentence = okt.morphs(sentence, stem=True)
            vector = tokenizer.texts_to_sequences(sentence)
            pad_new = pad_sequences(vector, maxlen=MAX_LENGTH)

            predictions = model.predict(pad_new)
            predictions = float(predictions.squeeze(-1)[0])
            
            if predictions > 0.5:
                print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
            else:
                print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))

            # num_chars_printed = len(sentence)

# def listen_print_loop(responses):
#     num_chars_printed = 0
#     for response in responses:
#         # print('여기서부터 시작: \n',response )
        
#     # while(True):
#         # responses.clear()
#         # response = responses[0]
#         if not response.results:
#             continue
#         result = response.results[0]
#         if not result.alternatives:
#             continue
#         #overwrite_chars = " " * (num_chars_printed - len(transcript))

#         if result.is_final:
#             # sys.stdout.write(transcript + overwrite_chars + "\r")
#             # sys.stdout.flush()

#             # num_chars_printed = len(transcript)
#             # print(response)
#             transcript = result.alternatives[0].transcript
#             print(transcript)
#             # print('전사:\n ' , transcript)
#             # print('전사:\n ' , transcript , '응답 : \n' , response,  '결과 :\n' , result)
#             transcript = okt.morphs(transcript, stem=True)
#             vector = tokenizer.texts_to_sequences(transcript)
#             pad_new = pad_sequences(vector, maxlen=MAX_LENGTH)

#             predictions = model.predict(pad_new)
#             predictions = float(predictions.squeeze(-1)[0])
            
#             if predictions > 0.5:
#                 print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
#             else:
#                 print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))
#             break

def main():
    language_code = "ko-KR"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results= True
        )

    with MicrophoneStream(RATE, CHUNK) as stream:   
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)
      
        # Now, put the transcription responses to use.
        listen_print_loop(responses)

if __name__ == "__main__":
    # init()
    main()
# while True:
    # sentence = input('단어 입력: ')
    # if sentence == '끝':
    #     break
    # sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]','', sentence)

    