import toml
from sklearn.datasets import make_classification
from sklearn import linear_model
import sys, os
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join('..', 'main')))
from OptimizerFactory import OptimizerFactory

if __name__=="__main__":
    config = toml.load("config/config.toml")
    optimizationConfig = config.get('Optimization')
    print(OptimizerFactory)
    optimizer = OptimizerFactory.get_optimizer(optimizationConfig)
    X, y = make_classification(n_samples=100, n_features=50, n_classes=3,
                            n_informative=4, n_redundant=1, n_repeated=2,
                            random_state=1)
    train_X,test_X, train_Y,test_Y = train_test_split( X, y, test_size=0.33, random_state=42)
    classifier = linear_model.LogisticRegression()
                            
    optimizer.run_trials(classifier, train_X,test_X,train_Y, test_Y, 50, mode="feature_selection")