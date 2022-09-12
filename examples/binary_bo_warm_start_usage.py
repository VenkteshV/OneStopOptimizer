import toml
from sklearn.datasets import make_classification
from sklearn import linear_model
import numpy as np
import sys, os
from sklearn.model_selection import train_test_split
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
    optimizer = OptimizerFactory.get_optimizer(optimizationConfig)
    X, y = make_classification(n_samples=1000, n_features=50, n_classes=3,
                            n_informative=4, n_redundant=1, n_repeated=2,
                            random_state=1)
    
    #this is not the right way to produce warm start data for feature selection. Random for convenience
    warm_X = X[:100]
    warm_Y = y[:100]
    X = X[100:]
    y = y[100:]
    train_X,test_X, train_Y,test_Y = train_test_split( X, y, test_size=0.33, random_state=42)
    for i in range(100):
        warm_X[i,:] = rand_bin_array(30,50)
    warm_X = warm_X.astype(int)
    classifier = linear_model.LogisticRegression()
                            
    optimizer.run_trials(classifier, train_X,test_X, train_Y,test_Y ,warm_X,warm_Y, 50, mode="feature_selection")