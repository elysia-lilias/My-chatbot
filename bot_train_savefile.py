#save word vector of input file (otherwise it waste a lot of time to run w2v every time I training)
import yaml
from collections import defaultdict
from datapre import word2vec
import numpy as np
import json

settings = {
    'window_size': 2,
    'n': 200,
    'epochs': 100,
    'learning_rate':0.1,
    'randepochs':0
}
dataname = "data_2.json"
max_seq_len = 10
limit = 10000000
def plot():
    from numpy import linalg as LA

    with open(dataname) as json_file:
        data = json.load(json_file)

    words = list(data.keys())
    v_count = len(words)
    n = settings['n']
    input = np.zeros((v_count, n))

    dim = n
    input = np.zeros((v_count, n))
    for i in range(v_count):
        word = words[i]
        input[i, :] = data[word]
    input = input.T
    Cov = 1 / dim * input.dot(input.T)
    w, v2 = LA.eig(Cov)
    wid = np.argsort(w)
    wid_1 = wid[dim - 1]
    wid_2 = wid[dim - 2]
    v = np.zeros((dim, 2))
    v[:, 0] = v2[:, wid_1]
    v[:, 1] = v2[:, wid_2]
    output = input
    output = v.T.dot(input)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(output[0,:],output[1,:])
    #maxx = np.max(output[0,:])
    #maxy = np.max(output[1, :])
    #minx = np.min(output[0, :])
    #miny = np.min(output[1, :])
    for i in range(v_count):
      ax.annotate(words[i],(output[0,i],output[1,i]))
    ax.axis('auto')
    ax.set_autoscale_on(True)
    #ax.plot([minx,maxx],[miny,maxy])
    plt.show()

def movieinput():
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import re
    import time
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

    for conv in convs:
        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])
    corpust = questions + answers
    corpus = [re.sub(r'[^A-Za-z0-9 -]+', '', word.lower()) + ' <EOS>' for word in corpust]
    return corpus

import yaml
with open("english/ai.yml", 'r') as stream:
    out = yaml.load(stream)

#k ="natural language processing and machine learning is fun and exciting"
with open('questions-word2s.txt') as f:
    lines = f.read().splitlines()
import re
question = [ out['conversations'][i][0] for i in range(len(out['conversations']))]
answer =  [ out['conversations'][i][1] for i in range(len(out['conversations']))]

#corpus = [re.sub(r'[^A-Za-z0-9 -]+', '', word.lower()) for word in lines]
corpust = question + answer
corpus = [re.sub(r'[^A-Za-z0-9 -]+', '', word.lower()) + ' <EOS>' for word in corpust]
corpus = movieinput()
#print(corpus)
w2v = word2vec(settings)
print("start training")

#training_data = w2v.generate_training_data(settings,corpus)

for rand in range(settings['randepochs']):
    w2v.generate_training_data(settings, corpus,rand)
###################################
    w2v.train()

corpus2 = [word.split() for word in corpus]
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import common_texts
#model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
import gensim.downloader as api

#corpuss = api.load('word2vec-google-news-300')  # download the corpus and return it opened as an iterable
#corpuss = api.load('text8')
model = Word2Vec(corpus2, min_count = 1, vector_size=200, window=5,  workers=4)
#model.train(corpus2, total_examples=443232,epochs=100)

#model.init_sims(replace=True)
dmax = np.amax(model.wv.vectors,axis = 0)
dmin = np.amin(model.wv.vectors,axis = 0)

model.wv.vectors = np.array([(model.wv.vectors[i,:] -dmin )/(dmax-dmin) for i in range(69687)])

#model.train(corpus2, total_examples= len(corpus2), epochs = 100)
with open("embeddings_norm.kv", "wb") as handle:
    model.wv.save(handle)
model.save("modelvc_norm.model")
#with open('data_withoutvector.json', 'w') as fp:
 #   json.dump(w2v.getonehot(), fp)




#with open('data_3.json', 'w') as fp:
#    json.dump(w2v.get_index(), fp)
#np.savetxt('w2vv.txt', w2v.get_w(), delimiter=',')
#plot()


#word = "machine"
#vec = w2v.word_vec(word)
#print(word, vec)

# 找相似的词
#w2v.vec_sim("machine", 3)


