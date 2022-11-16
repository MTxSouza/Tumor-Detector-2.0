# imports
from tensorflow.python.framework.ops import EagerTensor 
from tensorflow.keras.models import model_from_json

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os


class TrainLogging:
    
    def __init__(self) -> None:
        """Used to save training informations.
        """
        
        self.__model = os.path.join('../weights', 'model_' + str(os.listdir('../weights').__len__() + 1))
        self.__graphs = os.path.join(self.__model, 'graphs')
        
        os.mkdir(path=self.__model) # model folder
        os.mkdir(path=self.__graphs) # graphs folder
    
    def save(
        self,
        model: keras.engine.functional.Functional,
        version: str,
        best_loss: EagerTensor,
        best_val_loss: EagerTensor,
        accuracy: EagerTensor,
        val_accuracy: EagerTensor,
        loss_function: str,
        accuracy_function: str,
        optimizer: str,
        current_epoch: int,
        total_epoch: int,
        threshold: float,
        learning_rate: float,
        batch: int,
        loss_transformation: str,
        loss_graph: plt.Figure,
        validation_graph: plt.Figure,
        save_as_best: bool
        ) -> None:
        """Save all training informations in a specific model folder.

        Args:
            model (keras.engine.functional.Functional): Model to be saved.
            version (str): Model version.
            best_loss (EagerTensor): Best loss registered.
            best_val_loss (EagerTensor): Best val loss registered.
            accuracy (EagerTensor): Best accuracy registered.
            val_accuracy (EagerTensor): Best val loss registered.
            loss_function (str): Loss function name.
            accuracy_function (str): Accuracy function name.
            optimizer (str): Optimizer name.
            current_epoch (int): Current epoch of training.
            total_epoch (int): Total epoch of training.
            threshold (float): Threshold used to estimate accuracy.
            learning_rate (float): Learinig rate of training.
            batch (int): Batch of Dataset.
            loss_transformation (str): Transformation has done in loss values.
            loss_graph (plt.Figure): Figure of training loss.
            validation_graph (plt.Figure): Figure of validation graph.
            save_as_best (bool): Specify if the current savement is the best.
        """
        
        # checking variables
        assert isinstance(model, (keras.engine.functional.Functional)), 'Invalid type. model must be a keras.engine.functional.Functional'
        
        assert isinstance(version, (str)), 'Invalid type. version must be a str'
        
        assert isinstance(best_loss, (EagerTensor)), 'Invalid type. best_loss must be a EagerTensor'
        assert isinstance(best_val_loss, (EagerTensor)), 'Invalid type. best_val_loss must be a EagerTensor'
        
        assert isinstance(accuracy, (EagerTensor)), 'Invalid type. accuracy must be a EagerTensor'
        assert isinstance(val_accuracy, (EagerTensor)), 'Invalid type. val_accuracy must be a EagerTensor'
        
        assert isinstance(loss_function, (str)), 'Invalid type. loss_function must be a str'
        
        assert isinstance(accuracy_function, (str)), 'Invalid type. accuracy_function must be a str'
        
        assert isinstance(optimizer, (str)), 'Invalid type. optimizer must be a str'
        
        assert isinstance(current_epoch, (int)), 'Invalid type. current_epoch must be a int'
        
        assert isinstance(total_epoch, (int)), 'Invalid type. total_epoch must be a int'
        
        assert isinstance(threshold, (float)), 'Invalid type. threshold must be a float'
        
        assert isinstance(learning_rate, (float)), 'Invalid type. learning_rate must be a float'
        
        assert isinstance(batch, (int)), 'Invalid type. batch must be a int'
        
        assert isinstance(loss_transformation, (str)), 'Invalid type. loss_transformation must be a str'
        
        assert isinstance(loss_graph, (plt.Figure)), 'Invalid type. loss_graph must be a matplotlib.pyplot.Figure'
        
        assert isinstance(validation_graph, (plt.Figure)), 'Invalid type. validation_graph must be a matplotlib.pyplot.Figure'
        
        assert isinstance(save_as_best, (bool)), 'Invalid type. save_as_best must be a bool'
        
        # writting training parameters
        with open(os.path.join(self.__model, 'parameters.txt'), 'w') as parameters:
            parameters.writelines([
                f'{"-"*50}\n',
                f'Model Config: tumorNet_{version}.json\n',
                f'{"-"*50}\n',
                f'Epochs: {current_epoch}/{total_epoch}\n\n'
                f'Loss transformation: {loss_transformation}\n',
                f'Best loss: {best_loss.numpy()}\n',
                f'Best validation loss: {best_val_loss.numpy()}\n\n',
                f'Threshold: {threshold}\n',
                f'Accuracy: {accuracy}\n',
                f'Validation accuracy: {val_accuracy}\n\n',
                f'Loss Function: {loss_function}\n',
                f'Accuracy function: {accuracy_function}\n',
                f'Optimizer: {optimizer}\n',
                f'Learning Rate: {learning_rate}\n',
                f'Batch size: {batch}\n',
                f'{"-"*50}'
            ])
        
        # saving weights
        if save_as_best:
            model.save_weights(os.path.join(self.__model, 'checkpoints.h5'), save_format='h5')
            validation_graph.savefig(os.path.join(self.__graphs, 'best_validation_data.png'), bbox_inches='tight')
        
        # saving graphs
        loss_graph.savefig(os.path.join(self.__graphs, 'loss_graph.png'), bbox_inches='tight')
        validation_graph.savefig(os.path.join(self.__graphs, 'validation_data.png'), bbox_inches='tight')

def load_architecture(version: str) -> tf.Module:
    """Load the model architecture given a json file which contains
    the structure of model.

    Args:
        config (str): Path to the json file.
    """
    
    # checking variables
    assert isinstance(version, (str)), 'Invalid type. version must be a string'
    
    # setting up path
    version = os.path.join('../models', 'tumorNet_' + version + '.json')
    
    # checking if the file exists
    if os.path.exists(version):
        try:
            # reading file
            with open(version, 'r') as architecture:
                return model_from_json(architecture.read())
        except Exception as e:
            print(f"Error while reading the config file to load model. {e}.")
    else:
        raise FileExistsError("Invalid config path. Json file not found")