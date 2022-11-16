# imports
from sklearn.metrics import precision_recall_fscore_support,\
                            accuracy_score
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.backend import flatten,\
                                     sum
from tensorflow.keras.metrics import Metric
from typing import Union

import tensorflow as tf


"""
This scripts allows to calculate all available metrics for
segmentation task of Tumor Detector 2.0 project.
"""


class DiceCoeficient(Metric):
    
    def __init__(self, name='dice coeficient', dtype=None, **kwargs):
        """Compute the accuracy using Dice Coenfient formula.
        The idea behind is to estimate how good is the overlapping
        between the predicted and the real mask.
        To calculate it, the algorithm multiply by 2 the overlapping
        between the real and predicted value, and then, divide it
        by the sum of both images area.

        Args:
            name (str, optional): Name of loss function. Defaults to 'dice coeficient'.
            
            `update_state`
                y_true (EagerTensor): The real mask.
                y_pred (EagerTensor): The predicted mask.
        """
        self.__smooth = 1
        super().__init__(name, dtype, **kwargs)
        
    def update_state(self, y_true, y_pred):
        
        # checking variables
        assert isinstance(y_true, (EagerTensor)), 'Invalid type. y_true must be a EagerTensor'
        
        assert isinstance(y_pred, (EagerTensor)), 'Invalid type. y_pred must be a EagerTensor'
        
        # flatten values
        __y_true_flatten = flatten(y_true)
        __y_pred_flatten = flatten(y_pred)
        
        # intersection
        __intersection = sum(__y_true_flatten * __y_pred_flatten)
        
        self.__results = (2. * __intersection + self.__smooth) / (sum(__y_true_flatten) + sum(__y_pred_flatten) + self.__smooth)
    
    def result(self):
        return self.__results

class IoU(Metric):
    
    def __init__(self, name='intersection over union', dtype=None, **kwargs):
        """Compute the accuracy using IoU formula.
        The idea behind is to estimate how good is the overlapping
        between the predicted and the real mask.
        To calculate it, the algorithm calculate the overlapping
        between the real and predicted value, and then, divide it
        by the total area of both image.

        Args:
            name (str, optional): Name of accuracy function. Defaults to 'intersection over union'.
            
            `update_state`
                y_true (EagerTensor): The real mask.
                y_pred (EagerTensor): The predicted mask.
        """
        super().__init__(name, dtype, **kwargs)
        
    def update_state(self, y_true, y_pred):
        
        # checking variables
        assert isinstance(y_true, (EagerTensor)), 'Invalid type. y_true must be a EagerTensor'
        
        assert isinstance(y_pred, (EagerTensor)), 'Invalid type. y_pred must be a EagerTensor'
        
        # flatten values
        __y_true_flatten = flatten(y_true)
        __y_pred_flatten = flatten(y_pred)
        
        # intersection
        __intersection = sum(__y_true_flatten * __y_pred_flatten)
        
        # union
        __union = sum(__y_true_flatten) + sum(__y_pred_flatten) - __intersection
        
        self.__results = __intersection / __union
    
    def result(self):
        return self.__results


def overall_results(labels: list, pred_labels: list) -> Union[float, list, list]:
    """Calculate the Precision, Recall and Accuracy of model.
    It's necessary to estimate the label of image based on
    the predicted mask.

    Args:
        labels (list): List of labels. [0-1]
        pred_labels (list): List of predicted labels. [0-1]

    Returns:
        Union[float, list, list]: Overall accuracy, Recall, Precision.
    """
    
    # checking variables
    assert isinstance(labels, (list)), 'Invalid type. labels must be a list'
    
    assert isinstance(pred_labels, (list)), 'Invalid type. pred_labels must be a list'
    
    __precision, __recall, _, _ = precision_recall_fscore_support(y_true=labels, y_pred=pred_labels)
    return accuracy_score(y_true=labels, y_pred=pred_labels),\
            __recall,\
            __precision
            
def loss_menager(loss: EagerTensor, apply: str = 'real') -> EagerTensor:
    """Change the loss result. It allows to avoid either negative
    or weird loss values.

    Args:
        loss (EagerTensor): Loss of model.
        apply (str, optional): Changes to be done. Defaults to 'real'.
        - real (No changes), abs, negative.

    Returns:
        EagerTensor: The transformed loss.
    """
    
    # checking varaibles
    assert isinstance(loss, (EagerTensor)), 'Invalid type. loss must be a EagerTensor'
    
    assert isinstance(apply, (str)), 'Invalid type. apply must be a str'
    assert apply in ['real', 'abs', 'negative'], f'Invalid value. apply must be one of them: [real, abs, negative], it received {apply}'
    
    __transformer = {
        'real': loss,
        'abs': tf.abs(loss),
        'negative': -loss
    }
    
    return __transformer[apply]

def apply_threshold(mask: EagerTensor, threshold: float) -> EagerTensor:
    """Apply the threshold on mask image, rounding all
    pixels either to 0 or 1.

    Args:
        mask (EagerTensor): Mask to be modified.
        threshold (float): Threshold limit.

    Returns:
        EagerTensor: Mask with threshold applied.
    """
    
    # checking variables
    assert isinstance(mask, (EagerTensor)), 'Invalid type. mask must be a EagerTensor'
    
    assert isinstance(threshold, (float)), 'Invalid type. mask must be a float'
    
    # creating new copy of mask
    __new_mask = mask.numpy().copy()
    
    # applying threshold
    __new_mask[__new_mask >= threshold] = 1
    __new_mask[__new_mask < threshold] = 0
    
    return tf.constant(__new_mask)