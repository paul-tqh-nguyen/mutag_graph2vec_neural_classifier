
'''

This file contains global variables, global initializations, and some helper functions and data structures used across many modules.

Sections:
* Imports
* Logging
* Globals
* Visualize 2D PCA
* Vector Dict

'''

###########
# Imports #
###########

import os
import sys
import logging
import torch
import numpy as np
import matplotlib.cm
from contextlib import contextmanager
from sklearn.decomposition import PCA
from typing import Union, Iterable, Generator

from misc_utilities import *

###########
# Logging #
###########

LOGGER_NAME = 'mutag_logger'
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER_OUTPUT_FILE = './logs.txt'
LOGGER_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)

def _initialize_logger() -> None:
    LOGGER.setLevel(logging.INFO)
    logging_formatter = logging.Formatter('{asctime} - pid: {process} - threadid: {thread} - func: {funcName} - {levelname}: {message}', style='{')
    logging_file_handler = logging.FileHandler(LOGGER_OUTPUT_FILE)
    logging_file_handler.setFormatter(logging_formatter)
    LOGGER.addHandler(logging_file_handler)
    LOGGER.addHandler(LOGGER_STREAM_HANDLER)
    return

_initialize_logger()

@contextmanager
def training_logging_suppressed() -> Generator:
    logger_to_original_level = {}
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and 'lightning' == name:
            logger_to_original_level[logger] = logger.level
            logger.setLevel(logging.ERROR)
    with open(os.devnull, 'w') as dev_null:
        orignal_stream = LOGGER_STREAM_HANDLER.stream
        LOGGER_STREAM_HANDLER.setStream(dev_null)
        yield
        LOGGER_STREAM_HANDLER.setStream(orignal_stream)
    for logger, original_level in logger_to_original_level.items():
        logger.setLevel(original_level)
    return

###########
# Globals #
###########

RANDOM_SEED = 1234

EMBEDDING_VISUALIZATION_FILE_BASENAME = 'embedding_visualization.png'
CLASSIFICATION_CORRECTNESS_VISUALIZATION_FILE_BASENAME = 'classification_visualization.png'
KEYED_EMBEDDING_PICKLE_FILE_BASENAME = 'doc2vec_keyed_embedding.pickle'
DOC2VEC_MODEL_FILE_BASENAME = 'doc2vec.model'
RESULT_SUMMARY_JSON_FILE_BASENAME = 'result_summary.json'
    
MUTAG_CLASSIFIER_CHECKPOINT_DIR = './checkpoints_mutag_classifier'
MUTAG_CLASSIFIER_STUDY_NAME = 'classifier-mutag'
MUTAG_CLASSIFIER_DB_URL = 'sqlite:///classifier-mutag.db'

HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION = './docs/hyperparameter_search_results.json'
MUTAG_DATA_SUMMARY_JSON_FILE_LOCATION = './docs/mutag_data.json'

NUMBER_OF_MUTAG_CLASSIFIER_HYPERPARAMETER_TRIALS = 100_000
GPU_IDS = [0, 1, 2, 3]

if not os.path.isdir(MUTAG_CLASSIFIER_CHECKPOINT_DIR):
    os.makedirs(MUTAG_CLASSIFIER_CHECKPOINT_DIR)

ENABLE_VISUALIZATION_SAVING = False

####################
# Visualize 2D PCA #
####################

def visualize_vectors(matrix: np.ndarray, labels: np.ndarray, output_file_location: str, plot_title: str) -> None:
    if ENABLE_VISUALIZATION_SAVING:
        assert matrix.shape[0] == len(labels)
        pca = PCA(n_components=2, copy=False)
        pca.fit(matrix)
        matrix_transformed = pca.transform(matrix)
        with temp_plt_figure(figsize=(20.0,10.0)) as figure:
            plot = figure.add_subplot(111)
            plot.axvline(c='grey', lw=1, ls='--', alpha=0.5)
            plot.axhline(c='grey', lw=1, ls='--', alpha=0.5)
            label_to_color_map = matplotlib.cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
            label_to_color_map = dict(enumerate(label_to_color_map))
            colors = np.array([label_to_color_map[label] for label in labels])
            plot.scatter(matrix_transformed[:,0], matrix_transformed[:,1], c=colors, alpha=0.25)
            plot.set_title(plot_title)
            plot.set_xlabel('PCA 1')
            plot.set_ylabel('PCA 2')
            figure.savefig(output_file_location)
        LOGGER.info(f'Embeddings visualized at {embedding_visualization_location}')
    
###############
# Vector Dict #
###############

class VectorDict():
    '''Index into matrix by keys'''

    def __init__(self, keys: Iterable, matrix: Union[torch.tensor, np.ndarray]):
        assert len(matrix.shape) == 2
        assert len(keys) == matrix.shape[0]
        self.key_to_index_map = dict(map(reversed, enumerate(keys)))
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix)
        self.matrix = matrix

    def __getitem__(self, key: Union[int, torch.Tensor, Iterable]) -> np.ndarray:
        if isinstance(key, int):
            item = self.matrix[self.key_to_index_map[key]]
        else:
            item = self.matrix[[self.key_to_index_map[int(sub_key)] for sub_key in key]]
        return item

    def to(self, device_spec: Union[str, torch.device]) -> 'VectorDict':
        self.matrix.to(device_spec)
        return self

    def keys(self) -> Iterable:
        return self.key_to_index_map.keys()

    @property
    def shape(self) -> torch.Size:
        return self.matrix.shape
