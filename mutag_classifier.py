
'''

This module contains our MUTAG classifier

Sections:
* Imports
* Globals
* Graph Data Module
* MUTAG Classifier Model

'''

###########
# Imports #
###########

import karateclub
import json
import numpy as np
import networkx as nx
import pickle
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils import data
from collections import OrderedDict
from typing import Tuple, Dict
from typing_extensions import Literal

from global_values import *
from misc_utilities import *

###########
# Globals #
###########

BCE_LOSS = nn.BCELoss(reduction='none')

#####################
# Graph Data Module #
#####################

class MUTAGDataset(data.Dataset):
    def __init__(self, graph_id_to_graph_label: Dict[int, int]):
        self.graph_id_to_graph_label = OrderedDict(graph_id_to_graph_label)
        self.index_to_graph_id = np.array(list(self.graph_id_to_graph_label.keys()), dtype=int)
        
    def __getitem__(self, index: int):
        graph_id = self.index_to_graph_id[index]
        return {'graph_id': torch.tensor(graph_id, dtype=torch.float32), 'target': torch.tensor(self.graph_id_to_graph_label[graph_id], dtype=torch.float32)}
    
    def __len__(self):
        return len(self.graph_id_to_graph_label)

class MUTAGDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, graph_id_to_graph_label: Dict[int, int]): 
        self.batch_size = batch_size
        self.graph_id_to_graph_label = graph_id_to_graph_label
        
    def prepare_data(self) -> None:
        return
    
    def setup(self) -> None:
        
        graph_ids = list(self.graph_id_to_graph_label.keys())
        training_graph_ids, testing_graph_ids = train_test_split(graph_ids, test_size=0.20, random_state=RANDOM_SEED)
        validation_graph_ids, testing_graph_ids = train_test_split(testing_graph_ids, test_size=0.5, random_state=RANDOM_SEED)
        
        training_graph_id_to_graph_label = {}
        validation_graph_id_to_graph_label = {}
        testing_graph_id_to_graph_label = {}

        for graph_id, graph_label in self.graph_id_to_graph_label.items():
            if graph_id in training_graph_ids:
                training_graph_id_to_graph_label[graph_id] = graph_label
            elif graph_id in validation_graph_ids:
                validation_graph_id_to_graph_label[graph_id] = graph_label
            elif graph_id in testing_graph_ids:
                testing_graph_id_to_graph_label[graph_id] = graph_label
            else:
                raise ValueError(f'{graph_id} not in any split.')
        
        training_dataset = MUTAGDataset(training_graph_id_to_graph_label)
        validation_dataset = MUTAGDataset(validation_graph_id_to_graph_label)
        testing_dataset = MUTAGDataset(testing_graph_id_to_graph_label)

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/408 forces us to use shuffle in training and drop_last pervasively
        self.training_dataloader = data.DataLoader(training_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)
        self.validation_dataloader = data.DataLoader(validation_dataset, batch_size=len(validation_dataset)//4, num_workers=0, shuffle=False, drop_last=True)
        self.testing_dataloader = data.DataLoader(testing_dataset, batch_size=len(testing_dataset)//4, num_workers=0, shuffle=False, drop_last=True)
        
        assert 0 < len(self.training_dataloader) <= len(training_graph_ids) == len(training_graph_id_to_graph_label)
        assert 0 < len(self.validation_dataloader) <= len(validation_graph_ids) == len(validation_graph_id_to_graph_label)
        assert 0 < len(self.testing_dataloader) <= len(testing_graph_ids) == len(testing_graph_id_to_graph_label)
        
        return
    
    def train_dataloader(self) -> data.DataLoader:
        return self.training_dataloader

    def val_dataloader(self) -> data.DataLoader:
        return self.validation_dataloader

    def test_dataloader(self) -> data.DataLoader:
        return self.testing_dataloader

##########################
# MUTAG Classifier Model #
##########################

class MUTAGClassifier(pl.LightningModule):

    hyperparameter_names = (
        # graph2vec Hyperparameters
        'wl_iterations',
        'embedding_size',
        'graph2vec_epochs',
        'graph2vec_learning_rate',
        # NN Classifier Hyperparameters
        'batch_size', 
        'classifier_learning_rate',
        'number_of_layers',
        'gradient_clip_val',
        'dropout_probability',
    )
    
    def __init__(self, graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int], wl_iterations: int, embedding_size: int, graph2vec_epochs: int, graph2vec_learning_rate: float, batch_size: int, classifier_learning_rate: float, number_of_layers: int, gradient_clip_val: float, dropout_probability: float):
        super().__init__()
        self.save_hyperparameters(*(self.__class__.hyperparameter_names))
        
        self.linear_layers = nn.Sequential(
            OrderedDict( # @todo try decreasing the linear layer sizes
                sum(
                    [
                        [
                            (f'dense_layer_{i}', nn.Linear(embedding_size, embedding_size)),
                            (f'dropout_layer_{i}', nn.Dropout(self.hparams.dropout_probability)),
                            (f'activation_layer_{i}', nn.ReLU(True)),
                        ] for i in range(number_of_layers)
                    ], [])
            )
        )
        
        self.prediction_layers = nn.Sequential(
            OrderedDict([
                (f'reduction_layer', nn.Linear(embedding_size, 1)),
                (f'activation_layer', nn.Sigmoid()),
            ])
        )

        self.graph_id_to_graph_embeddings: VectorDict = self.create_graph2vec_embeddings(graph_id_to_graph, graph_id_to_graph_label)
    
    def create_graph2vec_embeddings(self, graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int]) -> VectorDict:
        
        checkpoint_directory = self.__class__.checkpoint_directory_from_hyperparameters(**{hyperparameter_name: getattr(self.hparams, hyperparameter_name) for hyperparameter_name in self.__class__.hyperparameter_names})
        if not os.path.isdir(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        
        saved_model_location = os.path.join(checkpoint_directory, DOC2VEC_MODEL_FILE_BASENAME)
        keyed_embedding_pickle_location = os.path.join(checkpoint_directory, KEYED_EMBEDDING_PICKLE_FILE_BASENAME)
        
        if os.path.isfile(saved_model_location):
            with open(keyed_embedding_pickle_location, 'rb') as file_handle:
                graph_id_to_graph_embeddings = pickle.load(file_handle)
        else:
        
            graphs = graph_id_to_graph.values()
            assert all(nx.is_connected(graph) for graph in graphs)
            assert not any(nx.is_directed(graph) for graph in graphs)
            assert all(list(range(graph.number_of_nodes())) == sorted(graph.nodes()) for graph in graphs)
            
            graph2vec_trainer = karateclub.Graph2Vec(
                wl_iterations=self.hparams.wl_iterations,
                dimensions=self.hparams.embedding_size,
                workers=1,
                epochs=self.hparams.graph2vec_epochs,
                learning_rate=self.hparams.graph2vec_learning_rate,
                min_count=0,
                seed=RANDOM_SEED,
            )
            graph2vec_trainer.fit([graph_id_to_graph[graph_id] for graph_id in graph_id_to_graph.keys()])
            graph_embedding_matrix: np.ndarray = graph2vec_trainer.get_embedding()
            assert tuple(graph_embedding_matrix.shape) == (len(graph_id_to_graph), self.hparams.embedding_size)
            
            graph_id_to_graph_embeddings = VectorDict(graph_id_to_graph.keys(), graph_embedding_matrix)
            
            with open(keyed_embedding_pickle_location, 'wb') as file_handle:
                pickle.dump(graph_id_to_graph_embeddings, file_handle)

            embedding_visualization_location = os.path.join(checkpoint_directory, EMBEDDING_VISUALIZATION_FILE_BASENAME)
            embedding_labels = np.array([graph_id_to_graph_label[graph_id] for graph_id in graph_id_to_graph_label.keys()])
            visualize_vectors(graph_embedding_matrix, embedding_labels, embedding_visualization_location, 'Embedding Visualization via PCA')
        
        return graph_id_to_graph_embeddings

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.shape[0]
        assert tuple(batch.shape) == (batch_size, self.hparams.embedding_size)

        transformed_batch = self.linear_layers(batch)
        assert tuple(transformed_batch.shape) == (batch_size, self.hparams.embedding_size)
        
        predictions = self.prediction_layers(transformed_batch).squeeze(1)
        assert tuple(predictions.shape) == (batch_size,)
        
        return predictions
    
    def backward(self, _trainer: pl.Trainer, loss: torch.Tensor, _optimizer: torch.optim.Optimizer, _optimizer_idx: int) -> None:
        del _trainer, _optimizer, _optimizer_idx
        loss.backward()
        return

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.classifier_learning_rate)
        return {'optimizer': optimizer}

    @property
    def device(self) -> Union[str, torch.device]:
        return only_one({parameter.device for parameter in self.parameters()})
    
    def _get_batch_loss(self, batch_dict: dict) -> torch.Tensor:
        batch = self.graph_id_to_graph_embeddings[batch_dict['graph_id']].to(self.device)
        target_predictions = batch_dict['target'].to(self.device)
        batch_size = only_one(target_predictions.shape)
        assert tuple(batch.shape) == (batch_size, self.hparams.embedding_size)
        assert tuple(target_predictions.shape) == (batch_size,)
        
        predictions = self(batch)
        assert tuple(predictions.shape) == (batch_size,)
        bce_loss = BCE_LOSS(predictions, target_predictions)
        return bce_loss
    
    def training_step(self, batch_dict: dict, _: int) -> pl.TrainResult:
        loss = self._get_batch_loss(batch_dict)
        result = pl.TrainResult(minimize=loss)
        return result

    def training_step_end(self, training_step_result: pl.TrainResult) -> pl.TrainResult:
        assert tuple(training_step_result.minimize.shape) == (self.hparams.batch_size,), f'Training loss has shape {only_one(training_step_result.minimize.shape)} (expected {self.hparams.batch_size}).'
        mean_loss = training_step_result.minimize.mean()
        result = pl.TrainResult(minimize=mean_loss)
        result.log('training_loss', mean_loss, prog_bar=True)
        return result
    
    def _eval_step(self, batch_dict: dict) -> pl.EvalResult:
        loss = self._get_batch_loss(batch_dict)
        assert len(loss.shape) == 1 
        result = pl.EvalResult()
        result.log('loss', loss)
        return result
    
    def _eval_epoch_end(self, step_result: pl.EvalResult, eval_type: Literal['validation', 'testing']) -> pl.EvalResult:
        loss = step_result.loss.mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{eval_type}_loss', loss)
        return result
        
    def validation_step(self, batch_dict: dict, _: int) -> pl.EvalResult:
        return self._eval_step(batch_dict)

    def validation_epoch_end(self, validation_step_results: pl.EvalResult) -> pl.EvalResult:
        return self._eval_epoch_end(validation_step_results, 'validation')

    def test_step(self, batch_dict: dict, _: int) -> pl.EvalResult:
        return self._eval_step(batch_dict)

    def test_epoch_end(self, test_step_results: pl.EvalResult) -> pl.EvalResult:
        return self._eval_epoch_end(test_step_results, 'testing')
    
    class PrintingCallback(pl.Callback):
    
        def __init__(self, checkpoint_callback: pl.callbacks.ModelCheckpoint):
            super().__init__()
            self.checkpoint_callback = checkpoint_callback
        
        def on_init_start(self, trainer: pl.Trainer) -> None:
            LOGGER.info('')
            LOGGER.info('Initializing trainer.')
            LOGGER.info('')
            return
        
        def on_train_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Model: ')
            LOGGER.info(model)
            LOGGER.info('')
            LOGGER.info(f'Training GPUs: {trainer.gpus}')
            for hyperparameter_name in sorted(model.hparams.keys()):
                LOGGER.info(f'{hyperparameter_name}: {model.hparams[hyperparameter_name]:,}')
            LOGGER.info('')
            LOGGER.info('Data:')
            LOGGER.info('')
            LOGGER.info(f'Training Batch Size: {trainer.train_dataloader.batch_size:,}')
            LOGGER.info(f'Validation Batch Size: {only_one(trainer.val_dataloaders).batch_size:,}')
            LOGGER.info('')
            LOGGER.info(f'Training Batch Count: {len(trainer.train_dataloader):,}')
            LOGGER.info(f'Validation Batch Count: {len(only_one(trainer.val_dataloaders)):,}')
            LOGGER.info('')
            LOGGER.info(f'Training Example Count: {len(trainer.train_dataloader)*trainer.train_dataloader.batch_size:,}')
            LOGGER.info(f'Validation Example Count: {len(only_one(trainer.val_dataloaders))*only_one(trainer.val_dataloaders).batch_size:,}')
            LOGGER.info('')
            LOGGER.info('Starting training.')
            LOGGER.info('')
            return
        
        def on_train_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Training complete.')
            LOGGER.info('')
            return
    
        def on_test_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Starting testing.')
            LOGGER.info('')
            LOGGER.info(f'Testing Batch Size: {only_one(trainer.test_dataloaders).batch_size:,}')
            LOGGER.info(f'Testing Example Count: {len(only_one(trainer.test_dataloaders))*only_one(trainer.test_dataloaders).batch_size:,}')
            LOGGER.info(f'Testing Batch Count: {len(only_one(trainer.test_dataloaders)):,}')
            LOGGER.info('')
            return
        
        def on_test_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
            LOGGER.info('')
            LOGGER.info('Testing complete.')
            LOGGER.info('')
            LOGGER.info(f'Best validation model checkpoint saved to {self.checkpoint_callback.best_model_path} .')
            LOGGER.info('')
            return
    
    @staticmethod
    def checkpoint_directory_from_hyperparameters(wl_iterations: int, embedding_size: int, graph2vec_epochs: int, graph2vec_learning_rate: float, batch_size: int, classifier_learning_rate: float, number_of_layers: int, gradient_clip_val: float, dropout_probability: float) -> str:
        checkpoint_directory = os.path.join(
            MUTAG_CLASSIFIER_CHECKPOINT_DIR, 
            f'wl_iterations_{int(wl_iterations)}_' \
            f'embed_{int(embedding_size)}_' \
            f'graph2vec_epochs_{int(graph2vec_epochs)}_' \
            f'graph2vec_lr_{graph2vec_learning_rate:.5g}_' \
            f'batch_{int(batch_size)}_' \
            f'classifier_lr_{classifier_learning_rate:.5g}_' \
            f'number_of_layers_{int(number_of_layers)}_' \
            f'gradient_clip_{gradient_clip_val}_' \
            f'dropout_{dropout_probability:.5g}'
        )
        return checkpoint_directory
    
    def visualize_classification(self, data_module: pl.LightningDataModule, graph_id_to_graph_label: Dict[int, int]) -> Tuple[float, float, float, float]:
        hyperparameter_dict = { hyperparameter_name: getattr(self.hparams, hyperparameter_name) for hyperparameter_name in self.__class__.hyperparameter_names }
        
        checkpoint_dir = self.__class__.checkpoint_directory_from_hyperparameters(**hyperparameter_dict)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        output_file_location = os.path.join(checkpoint_dir, CLASSIFICATION_CORRECTNESS_VISUALIZATION_FILE_BASENAME)

        training_graph_ids = [int(example['graph_id'].item()) for example in data_module.train_dataloader().dataset]
        validation_graph_ids = [int(example['graph_id'].item()) for example in data_module.val_dataloader().dataset]
        testing_graph_ids = [int(example['graph_id'].item()) for example in data_module.test_dataloader().dataset]
        training_accuracy = validation_accuracy = testing_accuracy = 0
        
        self.eval()
        correctness_vector: np.ndarray = np.empty(self.graph_id_to_graph_embeddings.matrix.shape[0], dtype=bool)
        true_label_vector: np.ndarray = np.empty(self.graph_id_to_graph_embeddings.matrix.shape[0], dtype=int)
        prediced_label_vector: np.ndarray = np.empty(self.graph_id_to_graph_embeddings.matrix.shape[0], dtype=int)
        for graph_id, graph_label in graph_id_to_graph_label.items():
            embedding = self.graph_id_to_graph_embeddings[graph_id]
            batch = embedding.unsqueeze(0).to(self.device)
            prediction = self.forward(batch).round()
            prediction_is_correct = graph_label == prediction
            index = self.graph_id_to_graph_embeddings.key_to_index_map[graph_id]
            true_label_vector[index] = graph_label
            prediced_label_vector[index] = prediction
            correctness_vector[index] = prediction_is_correct
            if prediction_is_correct:
                if graph_id in training_graph_ids:
                    training_accuracy += 1
                elif graph_id in validation_graph_ids:
                    validation_accuracy += 1
                elif graph_id in testing_graph_ids:
                    testing_accuracy += 1
                else:
                    raise ValueError(f'{graph_id} not in any split.')
        training_accuracy /= len(training_graph_ids)
        validation_accuracy /= len(validation_graph_ids)
        testing_accuracy /= len(testing_graph_ids)
        total_accuracy = correctness_vector.sum() / len(correctness_vector)

        if ENABLE_VISUALIZATION_SAVING:
            label0_color = 'red'
            label1_color = 'blue'
            
            pca = PCA(n_components=2, copy=False)
            pca.fit(self.graph_id_to_graph_embeddings.matrix)
            matrix_transformed = pca.transform(self.graph_id_to_graph_embeddings.matrix)
            with temp_plt_figure(figsize=(20.0,10.0)) as figure:
                plot = figure.add_subplot(111)
                plot.axvline(c='grey', lw=1, ls='--', alpha=0.5)
                plot.axhline(c='grey', lw=1, ls='--', alpha=0.5)
                true_label_colors = [label0_color if label==0 else label1_color for label in true_label_vector]
                plot.scatter(matrix_transformed[:,0], matrix_transformed[:,1], c=true_label_colors, alpha=1.0, marker='o')
                correctness_colors = [label0_color if correctness else label1_color for correctness in correctness_vector]
                plot.scatter(matrix_transformed[:,0], matrix_transformed[:,1], c=correctness_colors, alpha=1.0, marker='+')
                plot.set_title(f'Classification Accuracy {100*total_accuracy:.2g}%')
                plot.set_xlabel('PCA 1')
                plot.set_ylabel('PCA 2')
                figure.savefig(output_file_location)
            
            LOGGER.info(f'Classification correctness visualized at {output_file_location}')
            
        return training_accuracy, validation_accuracy, testing_accuracy, total_accuracy
    
    @classmethod
    def train_model(cls, gpus: List[int], **model_initialization_args) -> float:

        hyperparameter_dict = {
            hyperparameter_name: hyperparameter_value
            for hyperparameter_name, hyperparameter_value in model_initialization_args.items()
            if hyperparameter_name in cls.hyperparameter_names
        }
        
        checkpoint_dir = cls.checkpoint_directory_from_hyperparameters(**hyperparameter_dict)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'checkpoint_{epoch:03d}_{val_checkpoint_on}'),
                save_top_k=1,
                verbose=False,
                save_last=True,
                monitor='val_checkpoint_on',
                mode='min',
            )
        
        trainer = pl.Trainer(
            callbacks=[cls.PrintingCallback(checkpoint_callback)],
            auto_lr_find=True,
            early_stop_callback=pl.callbacks.EarlyStopping(
                monitor='val_checkpoint_on',
                min_delta=0.001,
                patience=5,
                verbose=False,
                mode='min',
                strict=True,
            ),
            min_epochs=10,
            gradient_clip_val=model_initialization_args.get('gradient_clip_val', 0),
            terminate_on_nan=True,
            gpus=gpus,
            distributed_backend='dp',
            deterministic=True,
            # precision=16, # not supported for data parallel (e.g. multiple GPUs) https://github.com/NVIDIA/apex/issues/227
            logger=pl.loggers.TensorBoardLogger(checkpoint_dir, name='checkpoint_model'),
            default_root_dir=checkpoint_dir,
            checkpoint_callback=checkpoint_callback,
        )
        
        model = cls(**model_initialization_args)
        
        data_module = MUTAGDataModule(hyperparameter_dict['batch_size'], model_initialization_args['graph_id_to_graph_label'])
        data_module.prepare_data()
        data_module.setup()
                
        trainer.fit(model, data_module)
        test_results = only_one(trainer.test(model, datamodule=data_module, verbose=False, ckpt_path=checkpoint_callback.best_model_path))
        best_validation_loss = checkpoint_callback.best_model_score.item()
        LOGGER.info(f'Testing Loss: {test_results["testing_loss"]}')
        
        training_accuracy, validation_accuracy, testing_accuracy, total_accuracy = model.visualize_classification(data_module, model_initialization_args['graph_id_to_graph_label'])
        
        output_json_file_location = os.path.join(checkpoint_dir, RESULT_SUMMARY_JSON_FILE_BASENAME)
        with open(output_json_file_location, 'w') as file_handle:
            json_dict = {
                'testing_loss': test_results['testing_loss'],

                'training_accuracy': training_accuracy,
                'validation_accuracy': validation_accuracy,
                'testing_accuracy': testing_accuracy,
                'total_accuracy': total_accuracy,
                
                'best_validation_loss': best_validation_loss,
                'best_validation_model_path': checkpoint_callback.best_model_path,
                
                'training_set_batch_size': data_module.training_dataloader.batch_size,
                'training_set_batch_count': len(data_module.training_dataloader),
                'validation_set_batch_size': data_module.validation_dataloader.batch_size,
                'validation_set_batch_count': len(data_module.validation_dataloader),
                'testing_set_batch_size': data_module.test_dataloader().batch_size,
                'testing_set_batch_count': len(data_module.test_dataloader()),
            }
            json_dict.update(hyperparameter_dict)
            json.dump(json_dict, file_handle, indent=4)
            LOGGER.info('Result Summary:')
            for result_summary_key, result_summary_value in json_dict.items():
                LOGGER.info(f'    {result_summary_key}: {result_summary_value}')
        
        return best_validation_loss
