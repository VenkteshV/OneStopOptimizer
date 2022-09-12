import toml
from sklearn.datasets import make_classification
from sklearn import linear_model
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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
    warm_X = np.zeros((100,4))
    warm_Y = np.zeros(100)
    print(warm_X.shape,warm_Y.shape)
    train_X,test_X, train_Y,test_Y = train_test_split( X, y, test_size=0.33, random_state=42)
    for i in range(100):
        warm_X[i,:] = rand_bin_array(30,4)
        warm_Y[i] = np.random.uniform(0,1,1)
    warm_X = warm_X.astype(int)
    classifiers = [linear_model.LogisticRegression(), SVC(gamma='auto',kernel='rbf',probability=True), RandomForestClassifier(max_depth=2, random_state=0),
                    MLPClassifier(random_state=1, max_iter=1000) ]
    for classifier in classifiers:
        classifier.fit(train_X, train_Y)
                            
    optimizer.run_trials(classifiers, train_X,test_X, train_Y,test_Y ,warm_X,warm_Y, 50, mode="model_selection", metric="macro_F1")