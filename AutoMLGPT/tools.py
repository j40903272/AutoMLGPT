import json
import pdb
from sklearn.model_selection import train_test_split
from typing import Optioanl, Union, List, Dict, Tuple


from AutoMLGPT.agent import coding_agent


from typing import Optional, Union, List, Dict, Tuple
from pydantic import Field
import ast
import astunparse
import logging
import pandas as pd


def prompts(name: str, description: str):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator



class LoadDataTool:
    globals: Optional[Dict] = dict()
    locals: Optional[Dict] = dict()

    def __init__(self, status):
        self.status = status
        
    @prompts(
        name="Load",
        description="This is a tool to load dataset."
        "Input should be a dataset name."
    )
    def inference(self, query: str):
        dataset = data_loader.select(query)
        
        data = dataset['train']
        label = dataset['label']
        X = data.drop(label, axis=1)
        y = data[label]
        name = dataset['name']
        
        self.status.init_data(name, X, y)
        
        return f"Load dataset {name} success !"



class E2EHyperOptTool:
    globals: Optional[Dict] = dict()
    locals: Optional[Dict] = dict()

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # self.locals = {"X": X, 'Y': Y,}

    @prompts(
        name="hyperopt",
        description="This is a tool for hyper parameter tuning."
        "Input should be a estimator name."
    )
    def inference(self, query: str):
        clf = self.construct_estimator(query)
        param_grid = self.construct_param_grid(clf)
        search_algo = self.get_search_algo(clf, param_grid)
        clff = self.construct_search_algo(search_algo)
        search = clff.fit(self.X_train, self.y_train)
        print(f"Best search score: {search.best_score_}")
        score = self.evaluate(clf, search)
        return f"Final score after hyper parameter tuning is {score}"
        
    def evaluate(self, clf, search):
        from sklearn.metrics import mean_squared_error
        clff = clf.__class__(**search.best_params_)
        clff.fit(self.X_train, self.y_train)
        y_pred = clff.predict(self.X_test)
        score = mean_squared_error(self.y_test, y_pred)
        print(f"Testing score: {score}")
        return score
    
    def construct_search_algo(self, search_algo):
        # construct search instance
        construct_search_prompt = f"""
        Give me two lines of python code.
        Frist, from lib import {search_algo.strip()}.
        Second, instatiate an instance with {search_algo.strip()}(clf, param_grid, random_state=0, n_iter=10) and assign to hyper_algo variable.
        """
        codes = coding_agent.run(construct_search_prompt)
        print(f"\nConstructing search algo:\n```\n{codes}\n```\n")
        res = self.execute(codes)
        if 'hyper_algo' not in self.locals:
            raise Exception(res)
        return self.locals['hyper_algo']

    def get_search_algo(self, clf, param_grid):
        search_algo_reason_prompt = f"""
        You are now performing a hyper parameter search on a machine learning model.
        Your goal is to decide the search algorithm base on model type and parameter space.


        The given model is a {clf.__class__.__name__}.
        The paramters grid is {param_grid}.
        The avaiable search algorithm is as follow:

        1. GridSearchCV: Exhaustive search over specified parameter values for an estimator.
        2. HalvingGridSearchCV: Search over specified parameter values with successive halving.
        3. ParameterGrid: Grid of parameters with a discrete number of values for each.
        4. ParameterSampler: Generator on parameters sampled from given distributions.
        5. RandomizedSearchCV: Randomized search on hyper parameters.
        6. HalvingRandomSearchCV: Randomized search on hyper parameters with successive halving.

        Think about which algorithm to use base on the model, parameter space and algorithm description.
        Tell me which algorithm you have choosen and why.
        List all the other algorithms you have choosen not to use and why.
        """
        
        reason = coding_agent.run(search_algo_reason_prompt)
        print(reason)

        search_algo = coding_agent.run(f"{reason}\n\nAccording to above, give me the algorithm name of the choosen algorithm in a single token.")
        print(f"\nDeciding search algo:\n{search_algo}\n")
        return search_algo

    def construct_param_grid(self, clf):
        param_grid_prompt = f"""
        You are now performing a hyper parameter search on a machine learning model.
        Your goal is to construct the paramter space to search.

        For example:
        The model is a LinearSVR.
        The parameters are {LinearSVR().get_params()}
        The parameter grid can be defined as:
        ```
        param_grid = {{'C': [1, 10, 100, 1000], 'tol': [0.001, 0.0001]}}
        ```
        Remeber to define the space only on important features. 


        Now.
        The given model is a {clf.__class__.__name__}.
        The paramters are {clf.get_params()}.
        The detail description about the parameters is as follow:
        {clf.__doc__}

        Construct the parameter grid you will use for hyper parameter in json format:
        """

        param_grid = coding_agent.run(param_grid_prompt)
        param_grid = json.loads(param_grid)
        print(f"\nConstruct params grid:\n{param_grid}\n")
        self.locals['param_grid'] = param_grid
        return param_grid
        
    def construct_estimator(self, estimator_name: str):
        # construct clf instance
        construct_estimator_prompt = f"Give me two lines of python code. Construct a {estimator_name.strip()} estimator and assign the estimator clf variable."
        codes = coding_agent.run(construct_estimator_prompt)
        print(f"\nConstruct estimator with following:\n```\n{codes}\n```\n")
        res = self.execute(codes)
        if 'clf' not in self.locals:
            raise Exception(res)
        return self.locals['clf']
        
    def execute(self, cmd: str):
        try:
            tree = ast.parse(cmd)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(astunparse.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = astunparse.unparse(module_end)  # type: ignore
            try:
                return eval(module_end_str, self.globals, self.locals)
            except Exception:
                exec(module_end_str, self.globals, self.locals)
                return ""
        except Exception as e:
            return str(e)

class pythonShellTool:

    @prompts(
        name="python_repl_ast",
        description="This is a python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "The output of the command should not be too long."
        "The dataset is a dataframe assign to a variable df"
    )
    def inference(self, query: str):
        try:
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(astunparse.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = astunparse.unparse(module_end)  # type: ignore
            try:
                return eval(module_end_str, self.globals, self.locals)[:1024]
            except Exception:
                exec(module_end_str, self.globals, self.locals)
                return ""
        except Exception as e:
            return str(e)