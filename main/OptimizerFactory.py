import pyswarms as ps
from optimizerInterface import OptimizerInterface
import OptimizerConstant
import toml
from algorithms.particle_swarm import BinaryParticleSwarm
from algorithms.binary_BO import BinaryBayesOptPredictor
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.binary_bo_warm_start import BinaryBayesOptWarmtStartPredictor
from sklearn.datasets import make_classification
from sklearn import linear_model

class OptimizerFactory:
    @staticmethod
    def get_optimizer(config):
        if config.get("OptimizerClass").lower() == OptimizerConstant.BINARY_PARTICLE_SWARM.lower():
            return  BinaryParticleSwarm()
        elif config.get("OptimizerClass").lower() == OptimizerConstant.BINARY_BAYES_OPT.lower():
            return BinaryBayesOptPredictor()

        elif config.get("OptimizerClass").lower() == OptimizerConstant.BINARY_BAYES_OPT_WARM_START.lower():
            return BinaryBayesOptWarmtStartPredictor()
        # elif config.get('OptimizerClass').lower() == OptimizerConstants.BAYES_OPT_WARM_START.lower():
        #     return BayesoptWarmStart()
        elif config.get('OptimizerClass').lower() == OptimizerConstant.SIMULATED_ANNEALING.lower():
            return SimulatedAnnealing()
        else:
            classname = config.get("OptimizerClass")
            if classname not in globals():
                raise ValueError("No implementation found for the custom data generator class specified: {}".format(classname))
            if OptimizerInterface.implementedBy(classname)==False:
                raise ValueError("custom data generator class specified, ie {}, doesn't correctly implement interface OptimizerInterface (in OptimizerInterface.py)".format(classname))
            #Note, currently, no params are passed into the custom class, add config to enable
            return globals()[classname]()

if __name__=="__main__":
    config = toml.load("examples/config/config.toml")
    optimizationConfig = config.get('Optimization')
    optimizer = OptimizerFactory.get_optimizer(optimizationConfig)
    X, y = make_classification(n_samples=100, n_features=50, n_classes=3,
                            n_informative=4, n_redundant=1, n_repeated=2,
                            random_state=1)
    classifier = linear_model.LogisticRegression()
                            
    optimizer.run_trials(classifier, X, y, 30, 100)

    # optimizer.run_trials(params,payload)