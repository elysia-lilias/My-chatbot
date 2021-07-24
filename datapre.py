from collections import defaultdict

import nltk
import ssl
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
#import wordsegment
from fc_net import TwoLayerNet
from fc_net_git import TwoLayerNet2
import re


class word2vec():


# Codes reference:
# Methods to get training data: https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281
# Methods for testing: https://blog.csdn.net/qq_34290470/article/details/102843763
# Training methods: deep learning class assignment 3&4
# Loss function: https://datascience.stackexchange.com/questions/28538/word2vec-softmax-function


  def __init__(self,settings):
      self.n = settings['n']
      self.epochs = settings['epochs']
      self.lr =  settings['learning_rate']
      self.window = settings['window_size']
      self.steps = 0
      self.set = False


  def word2onehot(self,word):
    word_vec = [0 for i in range(0, self.v_count)]
    word_index = self.word_index[word]
    word_vec[word_index] = 1
    return word_vec


  def generate_training_data(self, settings, corpus,rand):
      if(not self.set):
         word_counts = defaultdict(int)
         for row in corpus:
           for word in row.split():
               word_counts[word] += 1
         self.v_count = len(word_counts.keys())
         self.words_list = list(word_counts.keys())
         self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
         self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
         self.set = True
      self.training_data = []
      st = 1
      lens = len(corpus)
      print("start generating")
      print(lens)
      for sentenceidx in range(100*rand,100*(rand+1)):
          sentence = corpus[sentenceidx].split()
          print("processing ",sentenceidx," over ",lens," lines.")
          sent_len = len(sentence)
          for i, word in enumerate(sentence):
              w_target = self.word2onehot(sentence[i])
              w_context = []
              for j in range(i - self.window, i + self.window):
                  if j != i and j <= sent_len - 1 and j >= 0:
                      w_context.append(self.word2onehot(sentence[j]))
              tmp = [w_target, w_context]
              self.training_data.append([w_target, w_context])


  def getonehot(self):
      dict = defaultdict(int)
      for i in range(len(self.words_list)):
          word = self.words_list[i]
          idx = self.word_index[word]
          word_vec = [0 for i in range(0, self.v_count)]
          word_vec[idx] = 1
          dict[word] = word_vec
      return dict

  def train(self):
          training_data = self.training_data
          std = 1e-3
          model = TwoLayerNet(input_dim = self.v_count, hidden_dim = self.n, num_classes = self.v_count, weight_scale = std, lr = self.lr)
          model2 = TwoLayerNet2(input_dim=self.v_count, hidden_dim=self.n, num_classes=self.v_count, weight_scale=std, lr = self.lr)
          for i in range(self.epochs):
              loss = model.loss(training_data,self.v_count)
             # loss2 = model2.loss2(training_data,self.v_count)
              print('Epoch:', i, "Loss:", loss)
              #print('Epoch:', i, "Loss2:", loss2)
          self.w1 = model.getw1()
          self.w2 = model.getw2()




  def word_vec(self, word):
    """
    获取词向量
    通过获取词的索引直接在权重向量中找
    """

    w_index = self.word_index[word]
    v_w = self.w1[w_index]

    return v_w


  def vec_sim(self, word, top_n):
    """
    找相似的词
    """

    v_w1 = self.word_vec(word)
    word_sim = {}

    for i in range(self.v_count):
        v_w2 = self.w1[i]
        theta_sum = np.dot(v_w1, v_w2)

        # np.linalg.norm(v_w1) 求范数 默认为2范数，即平方和的二次开方
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_sum / theta_den

        word = self.index_word[i]
        word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

    for word, sim in words_sorted[:top_n]:
        print(word, sim)


  def get_w(self):
    w1 = self.w1
    return w1

  def get_index(self):
      retdict = self.word_index
      for wordind in range(len(self.words_list)):
          tmp = np.zeros((1,len(self.words_list)))
          tmp[0,wordind] = 1
          dtl = np.dot(tmp,self.w1).tolist()
          [dtl2] = dtl
          retdict[self.words_list[wordind]] = dtl2
      return retdict


