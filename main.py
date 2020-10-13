
'''

This file contains functionality for hyperparameter search over our MUTAG classifier's hyperparameter search space.

Sections:
* Imports
* Globals
* Data Processing
* MUTAG Classifier Hyperparameter Search
* Default Model
* Hyperparameter Search Result Analysis
* Driver

'''

###########
# Imports #
###########

import argparse
import json
import more_itertools
import joblib
import optuna
import pandas as pd
import multiprocessing as mp
import networkx as nx
from typing import Dict, Tuple

from misc_utilities import *
from global_values import *
from mutag_classifier import MUTAGClassifier

###########
# Globals #
###########

EDGES_FILE = './data/MUTAG_A.txt'
EDGE_LABELS_FILE = './data/MUTAG_edge_labels.txt'
NODE_LABELS_FILE = './data/MUTAG_node_labels.txt'
GRAPH_IDS_FILE = './data/MUTAG_graph_indicator.txt'
GRAPH_LABELS_FILE = './data/MUTAG_graph_labels.txt'
    
###################
# Data Processing #
###################

def process_data() -> Tuple[dict, dict]:
    with open(GRAPH_IDS_FILE, 'r') as graph_ids_file_handle:
        node_id_to_graph_id = dict(enumerate(map(int, graph_ids_file_handle.readlines()), start=1))
        graph_id_to_graph = {graph_id: nx.Graph() for graph_id in set(node_id_to_graph_id.values())}
        for node_id, graph_id in node_id_to_graph_id.items():
            graph_id_to_graph[graph_id].add_node(node_id)
    with open(NODE_LABELS_FILE, 'r') as node_labels_file_handle:
        node_labels_file_lines = node_labels_file_handle.readlines()
        assert len(node_labels_file_lines) == len(node_id_to_graph_id)
        for node_id, node_label in enumerate(map(int, node_labels_file_lines), start=1):
            graph_id = node_id_to_graph_id[node_id]
            graph = graph_id_to_graph[graph_id]
            graph.nodes[node_id]['node_label'] = node_label
    with open(EDGES_FILE, 'r') as edges_file_handle:
        edges_file_lines = edges_file_handle.readlines()
        split_lines = eager_map(lambda s: s.split(','), edges_file_lines)
        assert set(map(len, split_lines)) == {2}
        edges = map(lambda l: (int(l[0]), int(l[1])), split_lines)
    with open(EDGE_LABELS_FILE, 'r') as edge_labels_file_handle:
        edge_labels = map(int, edge_labels_file_handle.readlines())
    for (src_id, dst_id), edge_label in zip(edges, edge_labels):
        graph_id = node_id_to_graph_id[src_id]
        graph = graph_id_to_graph[graph_id]
        assert dst_id in graph.nodes
        graph.add_edge(src_id, dst_id, edge_label=edge_label)
    with open(GRAPH_LABELS_FILE, 'r') as graph_labels_file_handle:
        graph_id_to_graph_label = dict(enumerate(map(lambda label: 1 if label.strip()=='1' else 0, graph_labels_file_handle.readlines()), start=1))
        assert set(graph_id_to_graph_label.values()) == {0, 1}
        assert len(graph_id_to_graph_label) == 188
    graph_id_to_graph = {graph_id: nx.convert_node_labels_to_integers(graph) for graph_id, graph in graph_id_to_graph.items()}
    assert set(graph_id_to_graph.keys()) == set(graph_id_to_graph_label.keys())
    return graph_id_to_graph, graph_id_to_graph_label

##########################################
# MUTAG Classifier Hyperparameter Search #
##########################################

class MUTAGClassifierHyperParameterSearchObjective:
    def __init__(self, graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int], gpu_id_queue: object):
        # gpu_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classes are generated dynamically
        self.graph_id_to_graph = graph_id_to_graph
        self.graph_id_to_graph_label = graph_id_to_graph_label
        self.gpu_id_queue = gpu_id_queue

    def get_trial_hyperparameters(self, trial: optuna.Trial) -> dict:
        hyperparameters = {
            # graph2vec Hyperparameters
            'wl_iterations': int(trial.suggest_int('wl_iterations', 1, 20)),
            'embedding_size': int(trial.suggest_int('embedding_size', 256, 2048)),
            'graph2vec_epochs': int(trial.suggest_int('graph2vec_epochs', 10, 1024)),
            'graph2vec_learning_rate': trial.suggest_uniform('graph2vec_learning_rate', 1e-6, 1e-2),
            # NN Classifier Hyperparameters
            'batch_size': int(trial.suggest_int('batch_size', 1, 1)),
            'classifier_learning_rate': trial.suggest_uniform('classifier_learning_rate', 1e-6, 1e-2),
            'number_of_layers': int(trial.suggest_int('number_of_layers', 0, 0)),
            'gradient_clip_val': trial.suggest_uniform('gradient_clip_val', 1.0, 2.0),
            'dropout_probability': trial.suggest_uniform('dropout_probability', 0.0, 0.5),
        }
        assert set(hyperparameters.keys()) == set(MUTAGClassifier.hyperparameter_names)
        return hyperparameters
    
    def __call__(self, trial: optuna.Trial) -> float:
        gpu_id = self.gpu_id_queue.get()

        hyperparameters = self.get_trial_hyperparameters(trial)
        checkpoint_dir = MUTAGClassifier.checkpoint_directory_from_hyperparameters(**hyperparameters)
        LOGGER.info(f'Starting MUTAG classifier training for {checkpoint_dir} on GPU {gpu_id}.')
        
        try:
            with suppressed_output():
                with warnings_suppressed():
                    best_validation_loss = MUTAGClassifier.train_model(gpus=[gpu_id], graph_id_to_graph=self.graph_id_to_graph, graph_id_to_graph_label=self.graph_id_to_graph_label, **hyperparameters)
        except Exception as exception:
            if self.gpu_id_queue is not None:
                self.gpu_id_queue.put(gpu_id)
            raise exception
        if self.gpu_id_queue is not None:
            self.gpu_id_queue.put(gpu_id)
        return best_validation_loss

def get_number_of_mutag_classifier_hyperparameter_search_trials(study: optuna.Study) -> int:
    df = study.trials_dataframe()
    if len(df) == 0:
        number_of_remaining_trials = NUMBER_OF_MUTAG_CLASSIFIER_HYPERPARAMETER_TRIALS
    else:
        number_of_completed_trials = df.state.eq('COMPLETE').sum()
        number_of_remaining_trials = NUMBER_OF_MUTAG_CLASSIFIER_HYPERPARAMETER_TRIALS - number_of_completed_trials
    return number_of_remaining_trials

def load_hyperparameter_search_study() -> optuna.Study:
    return optuna.create_study(study_name=MUTAG_CLASSIFIER_STUDY_NAME, sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.NopPruner(), storage=MUTAG_CLASSIFIER_DB_URL, direction='minimize', load_if_exists=True)

def hyperparameter_search_study_df() -> pd.DataFrame:
    return load_hyperparameter_search_study().trials_dataframe()

def mutag_classifier_hyperparameter_search(graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int]) -> None:
    study = load_hyperparameter_search_study()
    number_of_trials = get_number_of_mutag_classifier_hyperparameter_search_trials(study)
    optimize_kawrgs = dict(
        n_trials=number_of_trials,
        gc_after_trial=True,
        catch=(Exception,),
    )
    with mp.Manager() as manager:
        gpu_id_queue = manager.Queue()
        more_itertools.consume((gpu_id_queue.put(gpu_id) for gpu_id in GPU_IDS))
        optimize_kawrgs['func'] = MUTAGClassifierHyperParameterSearchObjective(graph_id_to_graph, graph_id_to_graph_label, gpu_id_queue)
        optimize_kawrgs['n_jobs'] = len(GPU_IDS)
        with joblib.parallel_backend('multiprocessing', n_jobs=len(GPU_IDS)):
            with training_logging_suppressed():
                study.optimize(**optimize_kawrgs)
    return

#################
# Default Model #
#################

def train_default_model(graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int]) -> None:
    MUTAGClassifier.train_model(
        graph_id_to_graph=graph_id_to_graph,
        graph_id_to_graph_label=graph_id_to_graph_label,
        gpus=GPU_IDS,
        # graph2vec Hyperparameters
        wl_iterations=5,
        embedding_size=1024,
        graph2vec_epochs=500,
        graph2vec_learning_rate=1e-2,
        # NN Classifier Hyperparameters
        batch_size=1,
        classifier_learning_rate=1e-3,
        number_of_layers=1,
        gradient_clip_val=1.5,
        dropout_probability=0.25,
    )
    return

#########################################
# Hyperparameter Search Result Analysis #
#########################################

def analyze_hyperparameter_search_results() -> None:
    df = hyperparameter_search_study_df()
    df = df.loc[df.state=='COMPLETE']
    params_prefix = 'params_'
    assert set(MUTAGClassifier.hyperparameter_names) == {column_name[len(params_prefix):] for column_name in df.columns if column_name.startswith(params_prefix)}
    result_summary_dicts = []
    for row in df.itertuples():
        hyperparameter_dict = {hyperparameter_name: getattr(row, params_prefix+hyperparameter_name) for hyperparameter_name in MUTAGClassifier.hyperparameter_names}
        checkpoint_dir = MUTAGClassifier.checkpoint_directory_from_hyperparameters(**hyperparameter_dict)
        result_summary_file_location = os.path.join(checkpoint_dir, RESULT_SUMMARY_JSON_FILE_BASENAME)
        with open(result_summary_file_location, 'r') as f:
            result_summary_dict = json.load(f)
            result_summary_dict['duration_seconds'] = row.duration.seconds
        result_summary_dicts.append(result_summary_dict)
    with open(HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION, 'w') as f:
        json.dump(result_summary_dicts, f, indent=4)
    LOGGER.info(f'Hyperparameter result summary saved to {HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION} .')
    all_graph_data = {}
    graph_id_to_graph, graph_id_to_graph_label = process_data()
    for graph_id, graph in graph_id_to_graph.items():
        graph_label = graph_id_to_graph_label[graph_id]
        graph_data = nx.readwrite.json_graph.node_link_data(graph, {'source': 'source', 'target': 'target'})
        all_graph_data[graph_id] = {'graph': graph_data, 'graph_label': graph_label}
    with open(MUTAG_DATA_SUMMARY_JSON_FILE_LOCATION, 'w') as f:
        json.dump(all_graph_data, f, indent=4)
    LOGGER.info(f'MUTAG summary saved to {MUTAG_DATA_SUMMARY_JSON_FILE_LOCATION} .')
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-train-default-model', action='store_true', help='Train the default classifier.')
    parser.add_argument('-hyperparameter-search', action='store_true', help='Perform several trials of hyperparameter search for the MUTAG classifier.')
    parser.add_argument('-analyze-hyperparameter-search-results', action='store_true', help=f'Analyze completed hyperparameter search trials.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,map(bool,vars(args).values())))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.train_default_model:
        graph_id_to_graph, graph_id_to_graph_label = process_data()
        train_default_model(graph_id_to_graph, graph_id_to_graph_label)
    elif args.hyperparameter_search:
        graph_id_to_graph, graph_id_to_graph_label = process_data()
        mutag_classifier_hyperparameter_search(graph_id_to_graph, graph_id_to_graph_label)
    elif args.analyze_hyperparameter_search_results:
        analyze_hyperparameter_search_results()
    else:
        raise ValueError('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
