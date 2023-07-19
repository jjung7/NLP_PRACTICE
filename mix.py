from __future__ import division
import re
import json
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import keras
import time
import sys
import os
from google.cloud import speech
import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

okt = Okt()
tokenizer = Tokenizer()

prepro_configs = json.load(open('DATA/CLEAN_DATA/data_configs.json','r'))
with open('DATA/CLEAN_DATA/tokenizer.pickle', 'rb') as handle:
    word_vocab = pickle.load(handle)

prepro_configs['vocab'] = word_vocab
tokenizer.fit_on_texts(word_vocab)
model = keras.models.load_model('DATA/my_models/')
model.load_weights('DATA/DATA_OUT/cnn_classifier_kr/weights.h5')
MAX_LENGTH = 8
stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한'] # 불용어 추가할 것이 있으면 이곳에 추가

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

def listen_print_loop(responses):
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]','', transcript)
        sentence = okt.morphs(sentence, stem=True) # 토큰화
        sentence = [word for word in sentence if not word in stopwords] # 불용어 제거
        vector  = tokenizer.texts_to_sequences(sentence)
        pad_new = pad_sequences(vector, maxlen = MAX_LENGTH) # 패딩

        predictions = model.predict(pad_new)
        predictions = float(predictions.squeeze(-1)[0])

        if(predictions > 0.5):
            print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
        else:
            print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))
        sys.stdout.flush()
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        sys.stdout.write(transcript + overwrite_chars + "\r")
        time.sleep(3)

def main():
    language_code = "ko-KR"
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)

if __name__ == "__main__":
    main()
