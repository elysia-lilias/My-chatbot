import sys
import math

import keras
#import tflearn
import tensorflow as tf
from keras import regularizers
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
import chardet
import numpy as np
import struct



question_seqs = []
answer_seqs = []

max_w = 10
float_size = 4
wordvecname = "embeddings_colab.kv"
from gensim.models import Word2Vec, KeyedVectors
word_vector_dict = KeyedVectors.load(wordvecname)
word_vec_dim = 200
max_seq_len = 10
word_set = {}
limit = 10000
ep = 1000
wordvc =  Word2Vec.load("modelvc_colab.model")
data = KeyedVectors.load(wordvecname)
dmax = 1#np.max(data.vectors) # 1 : value for <pad> and <eos>
dmin = 0#np.min(data.vectors) # 0 : value for <go>
from keras import backend as K

def init_seq(input_file):
    """读取切好词的文本文件，加载全部词序列
    """


    ##############
    datatt = len(word_vector_dict['you'])
    word_vec_dim = datatt

    file_object = open(input_file, 'r')
    vocab_dict = {}
    while True:
        question_seq = []
        answer_seq = []
        line = file_object.readline()
        if line:
            line_pair = line.split('|')
            line_question = line_pair[0]
            line_answer = line_pair[1]
            for word in line_question.decode('utf-8').split(' '):
                if word_vector_dict.has_key(word):
                    question_seq.append(word_vector_dict[word])
            for word in line_answer.decode('utf-8').split(' '):
                if word_vector_dict.has_key(word):
                    answer_seq.append(word_vector_dict[word])
        else:
            break
        question_seqs.append(question_seq)
        answer_seqs.append(answer_seq)
    file_object.close()

def vector_sqrtlen(vector):
    len = 0
    for item in vector:
        len += item * item
    len = math.sqrt(len)
    return len

def vector_cosine(v1, v2):
    if len(v1) != len(v2):
        sys.exit(1)
    sqrtlen1 = vector_sqrtlen(v1)
    sqrtlen2 = vector_sqrtlen(v2)
    value = 0
    for item1, item2 in zip(v1, v2):
        value += item1 * item2
    return value / (sqrtlen1*sqrtlen2)


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true*10), K.round(y_pred*10)))

def vector2word(vector):
    max_cos = -10000
    match_word = ''
    for word in word_vector_dict:
        v = word_vector_dict[word]
        cosine = vector_cosine(vector, v)
        if cosine > max_cos:
            max_cos = cosine
            match_word = word
    return (match_word, max_cos)

def vector2word2(vector):
        max_cos = -10000
        match_word = ''
       # for word in data.index_to_key:
        #    v = data[word]
         #   cosine = vector_cosine(vector, v)
        #    if cosine > max_cos:
         #       max_cos = cosine
          #      match_word = word
        match_word = wordvc.wv.most_similar(positive=[vector], topn=1)
        match_vec = wordvc.wv.get_vector(match_word[0][0]).reshape((200,1))
        eos_vec = np.zeros((200,1))

        eosdist1 = np.linalg.norm(vector)
     #   eosdist = eosdist1/eosdist2
      #  if(match_word[0][1] <  eosdist):
      #       match_word[0] = ('<EOS>',eosdist)
        return (match_word, max_cos, eosdist1)

class MySeq2Seq(object):
    """
    思路：输入输出序列一起作为input，然后通过slick和unpack切分
    完全按照论文说的编码器解码器来做
    输出的时候把解码器的输出按照词向量的200维展平，这样输出就是(?,seqlen*200)
    这样就可以通过regression来做回归计算了，输入的y也展平，保持一致
    """
    def __init__(self, max_seq_len = 10, word_vec_dim = 200, input_file='./segment_result_lined.3000000.pair.less'):
        self.max_seq_len = max_seq_len
        self.word_vec_dim = word_vec_dim
        self.input_file = input_file
        self.setup = False
        self.vocabsize = 0

    def generate_trainig_data(self):
        from gensim.models import Word2Vec, KeyedVectors
        data = KeyedVectors.load(wordvecname)
        import re
        if not self.setup:
          lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
          conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
          id2line = {}
          for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]
          convs = []
          for line in conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
            convs.append(_line.split(','))
          questions = []
          answers = []
          data['<EOS>'] = np.zeros_like(data['you'])
          for conv in convs:
            for i in range(len(conv) - 1):
                questions.append(id2line[conv[i]])
                answers.append(id2line[conv[i + 1]])
          question = [re.sub(r'[^A-Za-z0-9 -]+', '', word.lower())  for word in questions]
          question_seqs = [[data[qword] for qword in question_line.split()] for question_line in
                        question]
          answer = [re.sub(r'[^A-Za-z0-9 -]+', '', word.lower())  for word in answers]
          answer_seqs = [[data[qword] for qword in answer_line.split()] for answer_line in answer]
          #self.setup = True
        x_data = []
        y_data = []
        t_data = []
        self.vocabsize = len(question_seqs)
        for i in range(len(question_seqs)):
        #for i in range(100):
            question_seq = question_seqs[i]
            answer_seq = answer_seqs[i]
            if len(question_seq) < self.max_seq_len and len(answer_seq) < self.max_seq_len and i<limit:
                sequence_x = list(question_seq) + [dmin*np.ones(self.word_vec_dim)] * (self.max_seq_len-len(question_seq))
                sequence_y2 = [dmax*np.ones(self.word_vec_dim)] + answer_seq
                sequence_t = answer_seq + [-dmax*np.ones(self.word_vec_dim)]
                sequence_y2 = sequence_y2 + [dmin * np.ones(self.word_vec_dim)] * (self.max_seq_len - len(answer_seq) - 1)
                sequence_t = sequence_t + [dmin * np.ones(self.word_vec_dim)] * (
                            self.max_seq_len - len(answer_seq) - 1)
                x_data.append(sequence_x)
                y_data.append(sequence_y2)
                t_data.append(sequence_t)
               # ttt11 = question[i]
               # ttt2 = answer[i]
               # ttt2
                #print "right answer"
                #for w in answer_seq:
                #    (match_word, max_cos) = vector2word(w)
                #    if len(match_word)>0:
                #        print match_word, vector_sqrtlen(w)

        return np.array(x_data), np.array(y_data),np.array(t_data)

    def generate_test(self,str):
        from gensim.models import Word2Vec, KeyedVectors
        data = KeyedVectors.load(wordvecname)
        ##############
        datatt = len(data['you'])
        self.word_vec_dim = datatt
        import re
        str =re.sub(r'[^A-Za-z0-9 -]+', '', str.lower())
        question_seq = [data[qword] for qword in str.split() if data.__contains__(qword)]

        X = []
        y = []
        tmp1 = question_seq
        if len(tmp1) < self.max_seq_len:
                #sequence_x =  list(tmp1) + [np.zeros(word_vec_dim)] * (self.max_seq_len - len(tmp1))
                sequence_x = list(question_seq) + [0* np.ones(self.word_vec_dim)] * (
                            self.max_seq_len - len(question_seq))
                # sequence_xy = list(tmp1) + [np.zeros(word_vec_dim)] * (self.max_seq_len - len(tmp1))
                sequence_y = [np.zeros(word_vec_dim)] * self.max_seq_len
                X.append(sequence_x)
        return np.array(X)

    def define_models(self,n_input, n_output, n_units):
        import keras
        from tensorflow.keras import layers
        # define training encoder

        encoder_inputs = layers.Input(shape=(None, n_input))

     #   encoder_batch = BatchNormalization()
     #   encoder_inputt =  encoder_batch(encoder_inputs)

        encoder =   layers.LSTM(n_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
       # encoder_outputs, state_h, state_c = encoder(encoder_inputs,initial_state=encoder_states)
       # encoder_states = [state_h, state_c]
        # define training decoder

        decoder_inputs = layers.Input(shape=(None, n_output))
        decoder_lstm = layers.LSTM(n_units, return_sequences=True, return_state=True)


        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
      #  decoder_dense = layers.Dense(n_output)
        decoder_dense  = keras.Sequential(
    [
        layers.Dense(n_output, activation="tanh", name="layer1"),
        layers.Dense(n_output, activation="tanh", name="layer2"),
        layers.Dense(n_output, activation="tanh", name="layer3"),
      #  layers.Dense(n_output, activation="tanh", name="layer4"),
     #   layers.Dense(n_output, activation="tanh", name="layer5"),
    #    layers.Dense(n_output, activation="tanh", name="layer6"),
     #   layers.Dense(n_output, activation="tanh", name="layer7"),
     #   layers.Dense(n_output, activation="tanh", name="layer8"),
    #    layers.Dense(n_output, activation="tanh", name="layer9"),

        layers.Dense(n_output, name="layerfin"),
    ]
    )

        decoder_outputs = decoder_dense(decoder_outputs)
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # define inference encoder
        encoder_model = keras.Model(encoder_inputs, encoder_states)
        # define inference decoder


        decoder_state_input_h = layers.Input(shape=(n_units,))
        decoder_state_input_c = layers.Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        # return all models
        return model, encoder_model, decoder_model




    def train(self):
        import matplotlib.pyplot as plt
        trainXY, trainY, traint = self.generate_trainig_data()
        model, infenc, infdec = self.define_models(word_vec_dim,word_vec_dim,word_vec_dim)
       # model = self.model(feed_previous=False)
        optimizer = keras.optimizers.Adam(lr=0.1)
        model.compile(optimizer=optimizer, loss='cosine_similarity', metrics=['accuracy'])
        history = model.fit([trainXY, trainY],traint,epochs=ep)
        model.save('./model/bot')
        infenc.save('./model/encoder')
        infdec.save('./model/decoder')
        print('saved')
        plt.plot(history.history['acc'])
        plt.title('0.1 lr')
        plt.ion()
        plt.pause(0.1)
        plt.show()
      #   model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
        # list all data in history
        return model

    def train2(self):
        import matplotlib.pyplot as plt
        trainXY, trainY, traint = self.generate_trainig_data()
        model, infenc, infdec = self.define_models(word_vec_dim, word_vec_dim, word_vec_dim)
        # model = self.model(feed_previous=False)
        optimizer = keras.optimizers.Adam(lr=0.0005)
        model.compile(optimizer=optimizer, loss='cosine_similarity', metrics=[soft_acc])
        history = model.fit([trainXY, trainY], traint, epochs=ep,shuffle=True )
        model.save('./model/botfin')
        infenc.save('./model/encoderfin')
        infdec.save('./model/decoderfin')
        print('saved')
        plt.plot(history.history['loss'])
        plt.title('model14')
        plt.show()
        #   model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
        # list all data in history
        return model

    def load(self):
        dependencies = {
            'soft_acc': soft_acc
        }
        from keras.models import load_model
        #model = self.model(feed_previous=True)
        self.model= load_model('./model/botfin',custom_objects=dependencies )
        self.infenc= load_model('./model/encoderfin',custom_objects=dependencies )
        self.infdec= load_model('./model/decoderfin',custom_objects=dependencies )
        print(self.model.summary())
        from keras.utils.vis_utils import plot_model
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


    def mypredict(self, source, n_steps, cardinality):
        # encode
        infenc = self.infenc
        infdec = self.infdec
        state = infenc.predict(source)
        # start of sequence input
        target_seq = np.array([1.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # collect predictions
        output = list()
        for t in range(n_steps):
            # predict next char
            yhat, h, c = infdec.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
            #target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        return np.array(output)
