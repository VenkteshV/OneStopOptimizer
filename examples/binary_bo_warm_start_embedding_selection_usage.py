import toml
from sklearn.datasets import make_classification
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import sys, os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import urllib

import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
model_sent_1 = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
model_sent_2 = SentenceTransformer('stsb-bert-base')
model_sent_3 = SentenceTransformer('stsb-distilbert-base')

sys.path.append(os.path.abspath(os.path.join('..', 'main')))
from OptimizerFactory import OptimizerFactory

def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr.astype(int)

if __name__=="__main__":
    config = toml.load("config/config.toml")
    optimizationConfig = config.get('Optimization')
    print(OptimizerFactory)
    optimizer = OptimizerFactory.get_optimizer(optimizationConfig)
    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = load_files("20newsbydate/20news-bydate-train",
     categories=categories, shuffle=True, random_state=42, encoding='utf-8', decode_error='ignore') 
    twenty_test= load_files("20newsbydate/20news-bydate-test",
     categories=categories, shuffle=True, random_state=42, encoding='utf-8', decode_error='ignore')

    train_X = twenty_train.data
    train_Y = twenty_train.target
    test_X = twenty_test.data
    test_Y = twenty_test.target

    print(len(train_X), train_Y)
    # this si not the right way to warm start u need previous embedding models selected and correspondign f score etc.
    warm_X = np.zeros((100,3))
    warm_Y = np.zeros(100)
    print(warm_X.shape,warm_Y.shape)

    for i in range(100):
        warm_X[i,:] = rand_bin_array(30,3)
        warm_Y[i] = np.random.uniform(0,1,1)
    warm_X = warm_X.astype(int)
    classifier = MLPClassifier(random_state=1, max_iter=500)
    emb_models = [model_sent_1,model_sent_2, model_sent_3]
    # train_X,test_X, train_Y,test_Y = train_test_split( X, y, test_size=0.33, random_state=42)
                            
    optimizer.run_trials(classifier, train_X,test_X, train_Y,test_Y, warm_X, warm_Y, max_iter=50, mode="embedding_selection", embedding_models =emb_models)