
import zope.interface
import numpy as np
from optimizerInterface import OptimizerInterface
from ax import optimize
import OptimizerConstant
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
import torch
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting
from sklearn.metrics import precision_recall_fscore_support


@zope.interface.implementer(OptimizerInterface)
class BinaryBayesOptWarmtStartPredictor:
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
            self.warm_start_X = None
            self.warm_start_y = None
            self.alpha=None
            self.mode = None
            self.metric=None
            self.embedding_models=None

    # TODO: make the below two methods common and inherit them
    def feature_selection_eval(self,params):
        mask = np.array(list(params.values()))
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
        mask = np.array(list(params.values()))
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
            return 0



    def embedding_selection_eval(self, params):
        mask = np.array(list(params.values()))
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
        return objective

    def run_trials(self, classifier, train_X, test_X,  train_Y, test_Y, warm_X, warm_Y, max_iter=30, mode="model_selection", metric = "micro_F1", embedding_models=None):
        strategy = GenerationStrategy(name="Sobol+GPEI", steps=[GenerationStep(model=Models.SOBOL, num_arms=max_iter/10),
            GenerationStep(model=Models.GPEI, num_arms=max_iter)])
        ax_client = AxClient(strategy)
        if embedding_models:
            self.embedding_models = embedding_models      

        self.classifier, self.train_X, self.train_Y, self.test_X, self.test_Y, self.warm_start_X, self.warm_start_y = classifier, train_X, train_Y, test_X, test_Y, warm_X, warm_Y
        self.alpha=0.5
        self.mode, self.metric = mode,metric
        if self.mode == "feature_selection":
            bayes_opt_params = []
            for i in range(train_X.shape[1]):
                bayes_opt_params.append({
                    "name": "x"+str(i),
                    "type":"range",
                    "bounds":[0,1]
                })
            ax_client.create_experiment(
            parameters=bayes_opt_params,
            minimize=True)

        elif self.mode == "model_selection":
            bayes_opt_params = []
            for i in range(len(classifier)):
                bayes_opt_params.append({
                    "name": "x"+str(i),
                    "type":"range",
                    "bounds":[0,1]
                })

            ax_client.create_experiment(
                parameters=bayes_opt_params,
                minimize=False)

        elif self.mode == "embedding_selection":
            bayes_opt_params = []
            for i in range(len(embedding_models)):
                bayes_opt_params.append({
                    "name": "x"+str(i),
                    "type":"range",
                    "bounds":[0,1]
                })
            ax_client.create_experiment(
                parameters=bayes_opt_params,
                minimize=False)

        warm_start_params = dict()
        for i in range(self.warm_start_X.shape[0]):
            for j in range(self.warm_start_X.shape[1]):
                warm_start_params["x"+str(j)] = int(self.warm_start_X[i][j])
            ax_client.attach_trial(warm_start_params)
            ax_client.complete_trial(i, self.warm_start_y[i])
            

        for i in range(max_iter):
                parameters, trial_index = ax_client.get_next_trial()
                # Local evaluation here can be replaced with deployment to external system.
                ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluation_function(parameters))
        best_parameters, best_values = ax_client.get_best_parameters()
        print("best_params", list(best_parameters.values()), best_values)
        return list(best_parameters.values())