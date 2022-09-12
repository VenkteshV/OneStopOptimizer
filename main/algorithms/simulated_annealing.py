import mlrose 
import numpy as np 
import random
import zope.interface
from optimizerInterface import OptimizerInterface
import torch
import OptimizerConstant
from sklearn.metrics import precision_recall_fscore_support


@zope.interface.implementer(OptimizerInterface)
class SimulatedAnnealing:
    def __init__(self,eval_function=None,params=None):
        if eval_function is not None:
            self.eval_function = eval_function
        if params is not None:
            self.params = params
        self.classifier = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.alpha=None
        self.mode = None
        self.metric = None
        self.embedding_models = None
  

    @staticmethod
    def random_shuffling(vector_of_points):
        """
        This function carries out random shuffling of column data to generate samples.
        Data in each of the columns  in the input array is shuffled separately, meaning that the rows of the resultant array will contain random samples from the sample space.
        Args:
            vector_of_points(NumPy Array): Array containing ordered points generated from stratification. Should usually be the output of the lhs_points_generation function. Each column self.number_of_samples elements.
        Returns:
            vector_of_points(NumPy Array): 2-D array containing the shuffled data. Should contain number_of_sample rows, with each row representing a potential random sample from within the sample space.
        """

        _, nf = np.shape(vector_of_points)
        for i in range(0, nf):
            z_col = vector_of_points[:, i]
            np.random.shuffle(z_col)
            vector_of_points[:, i] = z_col
        return vector_of_points
    @staticmethod
    def data_unscaling_minmax(x_scaled, x_min, x_max):
        """
        This function performs column-wise un-scaling on the a minmax-scaled input dataset.
            Args:
                x_scaled(NumPy Array): The input data set to be un-scaled. Data values should be between 0 and 1.
                x_min(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual minimum value for each column. Must contain same number of elements as the number of columns in x_scaled.
                x_max(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual maximum value for each column. Must contain same number of elements as the number of columns in x_scaled.
            Returns:
                unscaled_data(NumPy Array): A 2-D numpy array containing the scaled data, unscaled_data = x_min + x_scaled * (x_max - x_min)
            Raises:
                IndexError: Function raises index error when the dimensions of the arrays are inconsistent.
        """
        # Check if it can be evaluated. Will return index error if dimensions are wrong
        if x_scaled.ndim == 1:  # Check if 1D, and convert to 2D if required.
            x_scaled = x_scaled.reshape(len(x_scaled), 1)
        if (x_scaled.shape[1] != x_min.size) or (x_scaled.shape[1] != x_max.size):
            raise IndexError('Dimensionality problems with data for un-scaling.')
        unscaled_data = x_min + x_scaled * (x_max - x_min)
        return unscaled_data

    """ implementation insipired from idaes lhs implementation"""
    def latin_hypercube_sampling(self, params):
        num_samples = 1
        dim = len(params)
        sample_list_bounds = list()
        min_bounds = list()
        max_bounds = list()


        for param in params: 
            min_bounds.append(param["bounds"][0])
            max_bounds.append(param["bounds"][1])

        for index,min_value in enumerate(min_bounds):
            min_bounds[index] = 0 if min_value<0 else min_value

        sample_list_bounds.append(min_bounds)
        sample_list_bounds.append(max_bounds)

        sample_points_vector = np.zeros((num_samples, dim))
        for i in range(dim):
            min_limit =  min_bounds[i]
            max_limit = max_bounds[i]
            strata_size = 1 / num_samples
            var_samples = np.zeros((num_samples, 1))
            for i in range(num_samples):
                strata_lb = i * strata_size
                sample_point = strata_lb + (random.random() * strata_size)
                var_samples[i, 0] = int((sample_point * (max_limit - min_limit)) + min_limit)
                sample_points_vector[:, i] = var_samples[:, 0]
        bounds_array = np.zeros((2, len(sample_list_bounds[0]),))
        bounds_array[0, :] = np.array(sample_list_bounds[0])
        bounds_array[1, :] = np.array(sample_list_bounds[1])
        # generated_sample_points = self.random_shuffling(sample_points_vector)
        final_sample_list = self.data_unscaling_minmax(sample_points_vector, bounds_array[0, :], bounds_array[1, :])

        init_state = list()
        for element in final_sample_list[0]:
            init_state.append(int(element))

        return init_state

    def feature_selection_eval(self,params):
        mask = np.array(params)
        if np.count_nonzero(mask) == 0:
            feature_subset = self.train_X
            test_subset = self.test_X
        else:
            feature_subset = self.train_X[:,mask==1]
            test_subset = self.test_X[:,mask==1]

        self.classifier.fit(feature_subset, self.train_Y)

        mean_accur =  (self.classifier.predict(test_subset)==self.test_Y).mean()
        
        objective = (self.alpha * (1.0 - mean_accur) +  (1.0 - self.alpha) * (1 - (feature_subset.shape[1] / self.train_X.shape[1])))
        return objective
    def model_selection_eval(self,params):
        mask = np.array(params)
        classifiers = np.array(self.classifier)
        chosen_classifiers = classifiers[mask==1]
        all_predictions = []
        for chosen_class in chosen_classifiers:
            predictions = chosen_class.predict_proba(self.test_X)
            all_predictions.append(predictions)
        if len(all_predictions) > 0:
            if torch.is_tensor(all_predictions[0]):
                combined_predictions =  torch.mean(torch.stack(all_predictions),dim=0)
                combined_predictions = combined_predictions.cpu().numpy()   
            else:
                combined_predictions = np.mean(np.stack(all_predictions),axis=0)
            combined_predictions = np.argmax(combined_predictions,axis=1).flatten()
            if self.metric == OptimizerConstant.METRIC_MICRO:
                metrics = precision_recall_fscore_support(self.test_Y, combined_predictions, average="micro")
                return metrics[2]
            elif self.metric == OptimizerConstant.METRIC_MACRO:
                metrics = precision_recall_fscore_support(self.test_Y, combined_predictions, average="macro")
                return metrics[2]
            else:
                metrics = precision_recall_fscore_support(self.test_Y, combined_predictions, average="weighted")
                return metrics[2]
        else: 
            return 0.1


    def embedding_selection_eval(self, params):
        mask = np.array(params)
        models = self.embedding_models
        chosen_models_indices = (mask ==1)
        chosen_models_indices = np.where(chosen_models_indices)[0]
        chosen_models = [models[i] for i in chosen_models_indices]
        if len(chosen_models) >0:
            embeddings = []
            test_embeddings = []
            for chosen_model in chosen_models:
                sent_embedding = chosen_model.encode(self.train_X)
                test_sent_embedding = chosen_model.encode(self.test_X)
                embeddings.append(torch.tensor(sent_embedding, dtype=torch.float))
                test_embeddings.append(torch.tensor(test_sent_embedding, dtype=torch.float))
            combined_embeddings = torch.cat(embeddings, dim=1)

            #training
            self.classifier.fit(combined_embeddings, self.train_Y)

            #testing 
            test_combined_embeddings = torch.cat(test_embeddings, dim=1)
            predictions = self.classifier.predict(test_combined_embeddings)
            if self.metric == OptimizerConstant.METRIC_MICRO:
                metrics = precision_recall_fscore_support(self.test_Y, predictions, average="micro")
                return metrics[2]
            elif self.metric == OptimizerConstant.METRIC_MACRO:
                metrics = precision_recall_fscore_support(self.test_Y, predictions, average="macro")
                return metrics[2]
            else:
                metrics = precision_recall_fscore_support(self.test_Y, predictions, average="weighted")
                return metrics[2]  
        else:
            return 0

            
    def evaluation_function(self,params):
        if self.mode=="feature_selection":
            objective = self.feature_selection_eval(params)
        elif self.mode == "model_selection":
            objective = self.model_selection_eval(params)
        elif self.mode == "embedding_selection":
            objective = self.embedding_selection_eval(params)
        print("objective value",objective)
        return int(objective)

    def run_trials(self, classifier,  train_X, test_X,  train_Y, test_Y, max_iter=100, mode="model_selection", metric = "micro_F1", embedding_models= None):
        self.classifier, self.train_X, self.train_Y, self.test_X, self.test_Y = classifier, train_X, train_Y, test_X, test_Y
        self.alpha=0.5
        self.mode = mode
        self.metric = metric
        fitness_cust = mlrose.CustomFitness(self.evaluation_function)
        if embedding_models:
            self.embedding_models = embedding_models  

        if self.mode =="feature_selection":
            annealing_params = []
            for i in range(train_X.shape[1]):
                annealing_params.append({
                    "name": "x"+str(i),
                    "type":"range",
                    "bounds":[0,1]
                })
                param_length = train_X.shape[1]
                problem_cust = mlrose.DiscreteOpt(length = param_length, fitness_fn = fitness_cust, maximize = False, max_val = 2)

        elif self.mode == "model_selection":
            annealing_params = []
            for i in range(len(classifier)):
                annealing_params.append({
                    "name": "classifier"+str(i),
                    "type":"range",
                    "bounds":[0,1]
                })
                param_length = len(classifier)
                problem_cust = mlrose.DiscreteOpt(length = param_length, fitness_fn = fitness_cust, maximize = True, max_val = 2)

        elif self.mode == "embedding_selection":
            annealing_params = []
            for i in range(len(embedding_models)):
                annealing_params.append({
                    "name": "embedding"+str(i),
                    "type":"range",
                    "bounds":[0,1]
                })
                param_length = len(embedding_models)
                problem_cust = mlrose.DiscreteOpt(length = param_length, fitness_fn = fitness_cust, maximize = True, max_val = 2)

        init_state = self.latin_hypercube_sampling(annealing_params)
        schedule = mlrose.ExpDecay()
        best_state, best_fitness = mlrose.simulated_annealing(problem_cust, schedule = schedule,
                                                    max_iters = max_iter, 
                                                    init_state = init_state, random_state = 1)
        print("best state and values are:", best_state)