# imports
from tensorflow.python.framework.ops import EagerTensor
from IPython.display import clear_output
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import keras


"""
This scripts allows to visualize the results of model while training and
after it.
"""


class TrainingGraph:
    
    def __init__(
        self,
        loss_function: keras.losses,
        accuracy_function: keras.metrics,
        total_epochs: int,
        batch_size: int
        ) -> None:
        """This class allows to see in real time the training process
        of Tumor Detector 2.0 project.

        Args:
            loss_function (tf.keras.losses): The loss function used in training.
            accuracy_function (tf.keras.metrics): The accuracy function used in training.
            total_epochs (int): Total of epochs.
            batch_size (int): Batch of Dataset.
        """
        
        # checking variables
        assert isinstance(total_epochs, (int)), 'Invalid type. total_epochs must be an int'
        assert total_epochs > 0, 'Invalid value. total_epochs must be bigger than zero'
        
        assert isinstance(batch_size, (int)), 'Invalid type. batch_size must be an int'
        assert batch_size > 0, 'Invalid value. batch_size must be bigger than zero'
        
        # variables
        self.__loss_name = loss_function.name
        self.__accuracy_name = accuracy_function.name
        self.__ttl_epochs = total_epochs
        
        self.__train_color = '#004CD9'
        self.__val_color = '#1FC195'
        self.__line_thickness = 1
        
        # variables to preview graph
        self.__test_current_epoch = total_epochs - 1
        
        self.__test_loss = np.random.random(size=int(50 * self.__test_current_epoch))
        self.__test_val_loss = np.random.normal(size=int(20 * self.__test_current_epoch))
        
        self.__test_accuracy = np.random.random(size=int(50 * self.__test_current_epoch))
        self.__test_val_accuracy = np.random.normal(size=int(20 * self.__test_current_epoch))
        
        self.__mean_test_loss = self.__test_loss.mean()
        self.__mean_test_val_loss = self.__test_val_loss.mean()
        
        self.__mean_test_accuracy = self.__test_accuracy.mean()
        self.__mean_test_val_accuracy = self.__test_val_accuracy.mean()
        
        # configs
        plt.style.use('dark_background')
    
    def __structure__(self, loss, val_loss, accuracy, val_accuracy, mean_loss, mean_val_loss, mean_accuracy, mean_val_accuracy, current_epoch, samples=[]) -> Union[plt.Figure, plt.Figure]:
        
        # figures
        clear_output()
        __loss_fig, __ax = plt.subplots(ncols=2, figsize=(20,5))
        
        # main title
        __ax[0].set_ylabel(
            f'EPOCH: {current_epoch}/{self.__ttl_epochs} | Loss ({self.__loss_name}) - [Train: {mean_loss:.5f} - Val: {mean_val_loss:.5f}] | Accuracy ({self.__accuracy_name}) - [Train: {mean_accuracy:.5f} - Val: {mean_val_accuracy:.5f}]',
            fontsize=14,
            color='white',
            rotation=0,
            ha='left',
            position=(0,1.1)
        )
        
        # losses and accuracies plots
        for index, (data, titles) in enumerate(zip([[loss, val_loss], [accuracy, val_accuracy]], ['Loss Graph', 'Accuracy Graph'])):
            
            for values, label, colors in zip(data, ['Train', 'Validation'], [self.__train_color, self.__val_color]):
                try:
                    __ax[index].plot(
                        np.arange(start=0, stop=current_epoch, step=current_epoch / values.__len__()),
                        values,
                        color=colors,
                        linewidth=self.__line_thickness,
                        label=label
                    )
                except:
                    __ax[index].plot(
                        np.arange(start=0, stop=current_epoch, step=current_epoch / (values.__len__() - 0.5)),
                        values,
                        color=colors,
                        linewidth=self.__line_thickness,
                        label=label
                    )
            __ax[index].set_xticks(np.arange(start=0, stop=current_epoch+1, step=1))
        
            # plot info
            __ax[index].set_title(
                titles,
                fontsize=10,
                color='white'
            )
        
            __ax[index].set_xlabel(
                'Epochs',
                fontsize=10,
                color='white'
            )
        
            # styles
            __ax[index].tick_params(
                axis='both',
                colors='gray',
                length=2,
                labelsize=8
            )
        
        # legend of graph
        plt.legend()
        
        # samples
        __n_samples = samples.__len__()
        __nrows = 12
        
        __samples_fig, __ax_samples = plt.subplots(nrows=__nrows, ncols=5, figsize=(22,22))
        for i, txt in enumerate(['Tomography Image', 'Real Tumor Mask', 'Predicted Tumor Mask\nThreshold: 0.3', 'Predicted Tumor Mask\nThreshold: 0.5', 'Predicted Tumor Mask\nThreshold: 0.7']):
            __ax_samples[0][i].set_title(txt, fontsize=12)
        
        if __n_samples:

            indexes = np.random.choice(a=np.arange(__n_samples), size=__nrows, replace=False)

            samples = samples[indexes]
            for i, data in enumerate(samples):

                # image
                __ax_samples[i][0].imshow(data[0], cmap='gray')

                # mask
                __ax_samples[i][1].imshow(data[0], cmap='gray')
                __ax_samples[i][1].imshow(data[1], cmap='gray', alpha=0.45)
                
                # predicted masks
                __ax_samples[i][2].imshow(data[0], cmap='gray')
                __ax_samples[i][2].imshow(data[2], cmap='gnuplot', alpha=0.45)
                
                __ax_samples[i][3].imshow(data[0], cmap='gray')
                __ax_samples[i][3].imshow(data[3], cmap='gnuplot', alpha=0.45)
                
                __ax_samples[i][4].imshow(data[0], cmap='gray')
                __ax_samples[i][4].imshow(data[4], cmap='gnuplot', alpha=0.45)
                
                for j in range(5):
                    __ax_samples[i][j].get_xaxis().set_visible(False)
                    __ax_samples[i][j].get_yaxis().set_visible(False)

        else:
            
            # image for preview
            preview_image = np.zeros(shape=[256,256], dtype='float32')
            
            for i in range(__nrows):
                
                # image
                __ax_samples[i][0].imshow(preview_image, cmap='gray')

                # mask
                __ax_samples[i][1].imshow(preview_image, cmap='gray')
                __ax_samples[i][1].imshow(preview_image, cmap='gray', alpha=0.45)
                
                # predicted masks
                __ax_samples[i][2].imshow(preview_image, cmap='gray')
                __ax_samples[i][2].imshow(preview_image, cmap='gnuplot', alpha=0.45)
                
                __ax_samples[i][3].imshow(preview_image, cmap='gray')
                __ax_samples[i][3].imshow(preview_image, cmap='gnuplot', alpha=0.45)
                
                __ax_samples[i][4].imshow(preview_image, cmap='gray')
                __ax_samples[i][4].imshow(preview_image, cmap='gnuplot', alpha=0.45)
                
                for j in range(5):
                    
                    __ax_samples[i][j].get_xaxis().set_visible(False)
                    __ax_samples[i][j].get_yaxis().set_visible(False)
        
        plt.show()
        
        return __loss_fig, __samples_fig
    
    def display(self) -> None:
        """Display a preview of training graph before start to train.
        """
        self.__structure__(self.__test_loss, self.__test_val_loss, self.__test_accuracy, self.__test_val_accuracy, self.__mean_test_loss, self.__mean_test_val_loss, self.__mean_test_accuracy, self.__mean_test_val_accuracy, self.__test_current_epoch)
    
    def plot(self, loss: list, val_loss: list, accuracy:list, val_accuracy: list, mean_loss: EagerTensor, mean_val_loss: EagerTensor, mean_accuracy: EagerTensor, mean_val_accuracy: EagerTensor, current_epoch: int, samples: np.ndarray) -> None:
        """Display the training graph.

        Args:
            loss (list): List of all losses of train data.
            val_loss (list): List of all losses of val data.
            accuracy (list): List of all accuracies of train data.
            val_accuracy (list): List of all accuracies of val data.
            mean_loss (EagerTensor): Mean train loss of current epoch.
            mean_val_loss (EagerTensor): Mean val loss of current epoch.
            mean_accuracy (EagerTensor): Mean train accuracy of current epoch.
            mean_val_accuracy (EagerTensor): Mean val accuracy of current epoch.
            current_epoch (int): Current epoch of training.
            samples (np.ndarray): Array of list, which contains [image, mask, predicted] data to be displyed.
        """
        
        # checking variables
        assert isinstance(loss, (list)), 'Invalid type. loss must be a list'
        
        assert isinstance(val_loss, (list)), 'Invalid type. val_loss must be a list'
        
        assert isinstance(accuracy, (list)), 'Invalid type. accuracy must be a list'
        
        assert isinstance(val_accuracy, (list)), 'Invalid type. val_accuracy must be a list'
        
        assert isinstance(mean_loss, (EagerTensor)), 'Invalid type. mean_loss must be a EagerTensor'
        
        assert isinstance(mean_val_loss, (EagerTensor)), 'Invalid type. mean_val_loss must be a EagerTensor'
        
        assert isinstance(mean_accuracy, (EagerTensor)), 'Invalid type. mean_accuracy must be a EagerTensor'
        
        assert isinstance(mean_val_accuracy, (EagerTensor)), 'Invalid type. mean_val_accuracy must be a EagerTensor'
        
        assert isinstance(current_epoch, (int)), 'Invalid type. current_epoch must be a float'
        
        assert isinstance(samples, (np.ndarray)), 'Invalid type. samples must be a np.ndarray'
        
        return self.__structure__(loss, val_loss, accuracy, val_accuracy, mean_loss, mean_val_loss, mean_accuracy, mean_val_accuracy, current_epoch, samples)

def apply_masks(image: np.ndarray, mask: np.ndarray, predicted_mask: np.ndarray) -> np.ndarray:
    """Apply both real mask and predicted mask into
    the image.

    Args:
        image (np.ndarray): Tomography image.
        mask (np.ndarray): Real mask.
        predicted_mask (np.ndarray): Predicted mask.

    Returns:
        np.ndarray: The tomography image with the real mask and the predicted mask displayed.
    """

    # applying three channels in mask
    __background = np.zeros(shape=[image.shape[0], image.shape[1], 3])
    __new_image = __background.copy()
    for channel in range(3):
        __new_image[:,:,channel] = image
    
    # applying real mask channel
    __background[:,:,0] = mask
    
    # applying predicted channel
    __background[:,:,1] = predicted_mask

    __background = __new_image * __background
    __background = np.clip(__background, 0, 255)

    __new_image[__background>0] = __background[__background>0]

    return __new_image.astype('uint8')