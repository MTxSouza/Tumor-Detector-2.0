# imports
from TumorDetector2.utils.metrics import overall_results,\
                                         apply_threshold
from TumorDetector2.utils.data import _tfrecord_reader

from tensorflow._api.v2.config import set_visible_devices,\
                                      list_physical_devices
from tensorflow.keras.models import model_from_json
from PIL import ImageTk, Image
from io import BytesIO
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
import argparse
import os


# configurations
SIZES = {
    0: {
        'window_size': '750x500',
        'icon_size': (50,50),
        'text_size': 10,
        'image_size': (400,400)
    },
    1: {
        'window_size': '1050x700',
        'icon_size': (65,65),
        'text_size': 12,
        'image_size': (600,600)
    },
    2: {
        'window_size': '1400x950',
        'icon_size': (80, 80),
        'text_size': 14,
        'image_size': (800,800)
    }
}
BACKGROUND_COLOR = '#272727'
STYLE_COLOR_1 = '#00D3FD'
STYLE_COLOR_2 = '#00D1AE'
TEXT_COLOR_1 = '#FFFFFF'
TEXT_COLOR_2 = '#000000'



# main screen
class Screen(tk.Tk):

    def __init__(self, window_size: int, version: str, on_cpu: bool = False, *args, **kwargs) -> None:
        """Display an interface has made by BRAIN (Brazilian Artificial Inteligence Nucleus)
        entirely in Python. It allows you to see the Pre-Trained model results in a fancy
        interable screen.

        Args:
            window_size (int): Size of image.
            version (str): Version of model to test.
            on_cpu (bool, optional): Force to run on CPU.
        """
        tk.Tk.__init__(self, *args, **kwargs)

        # checking variables
        assert isinstance(window_size, (int)), 'Invalid type. window_size must be an int'
        assert window_size in SIZES.keys(), f'Invalid value. window_size has to be one of them: {SIZES.keys()}'
        
        assert isinstance(on_cpu, (bool)), 'Invalid type. on_cpu must be a bool'

        # variables
        self.__current_index = 0
        plt.style.use('dark_background')

        # changing device
        if on_cpu:
            set_visible_devices([], 'GPU')
        else:
            gpus = list_physical_devices(device_type='GPU')
            if gpus.__len__():
                set_visible_devices(devices=gpus[0], device_type='GPU')
            else:
                raise RuntimeError('No GPU has been detected')

        # loading model and images
        __images = tf.data.TFRecordDataset(filenames=os.path.join('TumorDetector2', 'data', 'test.tfrecord')).map(_tfrecord_reader).batch(1).shuffle(True)
        
        __weights = os.path.join('TumorDetector2/models', 'tumorNet_' + version + '.h5')
        version = os.path.join('TumorDetector2/models', 'tumorNet_' + version + '.json')
        
        # checking if the file exists
        if os.path.exists(version):
            try:
                # reading file
                with open(version, 'r') as architecture:
                    __model = model_from_json(architecture.read())
                # loading weights
                try:
                    __model.load_weights(__weights)
                except Exception as e:
                    raise FileExistsError(f"Error while loading the weights. {e}.")
            except Exception as e:
                raise FileExistsError(f"Error while reading the config file to load model. {e}.")
        else:
            raise FileExistsError(f"Invalid config path. Json file not found: {version}")
        
        self.__data = []
        self.__labels = []
        self.__pred_labels = []
        
        # inference
        for __image, __mask, __label in tqdm(iterable=__images, desc='Running inferences'):
            
            # inference
            __pred = tf.constant(__model.predict(__image))
            
            # applying threshold
            __pred = apply_threshold(__pred, 0.5)
            
            # getting data as numpy
            __pred = __pred[0][:,:,0].numpy()
            __image = __image[0][:,:,0].numpy()
            __mask = __mask[0][:,:,0].numpy()
            
            # saving labels
            self.__pred_labels.append(1 if __pred.sum() else 0)
            self.__labels.append(__label.numpy()[0])
            
            # applying mask on image
            with BytesIO() as __buffer:
                __fig, __ax = plt.subplots(ncols=2)
                
                __ax[0].imshow(__image, cmap='gray')
                __ax[0].imshow(__mask, cmap='gray', alpha=0.45)
                __ax[0].set_title('Real')
                
                __ax[1].imshow(__image, cmap='gray')
                __ax[1].imshow(__pred, cmap='gnuplot', alpha=0.45)
                __ax[1].set_title('TumorNet')
                
                for index in range(2):
                    __ax[index].get_xaxis().set_visible(False)
                    __ax[index].get_yaxis().set_visible(False)
                
                __fig.savefig(fname=__buffer, format='jpeg')
                
                # loading image as pillow
                __data = Image.open(fp=__buffer).resize(SIZES[window_size]['image_size'])
                plt.close(fig=__fig)
                
            self.__data.append(ImageTk.PhotoImage(image=__data))
        self.__n_elements = self.__data.__len__()
        # calculating results
        accuracy, recall, precision = overall_results(self.__labels, self.__pred_labels)
        # getting data to be displayed
        self.__current_image = self.__data[self.__current_index]

        # setting up configurations of main screen
        # title
        self.wm_title(string='Tumor Detector 2.0 - Application')
        # size
        self.geometry(newGeometry=SIZES[window_size]['window_size'])
        self.resizable(width=False, height=False)
        # background color
        self.config(background=BACKGROUND_COLOR)

        # setting up configurations of image frame
        self.__image_frame = tk.Label(master=self, image=self.__current_image)
        # style
        self.__image_frame.config(
            background='black',
            highlightthickness=2,
            highlightbackground=STYLE_COLOR_1
        )
        # localization
        self.__image_frame.place(relx=0.01, rely=0.1, relwidth=0.6, relheight=0.85)
        # -- buttons
        button_next = tk.Button(master=self.__image_frame)
        button_previous = tk.Button(master=self.__image_frame)
        # -- buttons style
        button_previous.config(
            text='<<',
            fg=TEXT_COLOR_2,
            background=STYLE_COLOR_2,
            font=('Arial', SIZES[window_size]['text_size'], 'bold'),
            command=self.previous_next
        )
        button_next.config(
            text='>>',
            fg=TEXT_COLOR_2,
            background=STYLE_COLOR_2,
            font=('Arial', SIZES[window_size]['text_size'], 'bold'),
            command=self.next_image
        )
        # -- buttons localization
        button_next.place(relx=0.94, rely=0.94, relwidth=0.05, relheight=0.05)
        button_previous.place(relx=0.01, rely=0.94, relwidth=0.05, relheight=0.05)

        # setting up configurations of settings frame
        self.__informations_frame = tk.Frame(master=self)
        # style
        self.__informations_frame.config(
            background=BACKGROUND_COLOR,
            highlightthickness=2,
            highlightbackground=STYLE_COLOR_1
        )
        # localization
        self.__informations_frame.place(relx=0.65, rely=0.1, relwidth=0.3, relheight=0.85)
        # text
        self.__informations_text_1 = tk.Label(master=self.__informations_frame)
        # - style
        self.__informations_text_1.config(
            text='Informations',
            fg=TEXT_COLOR_1,
            background=BACKGROUND_COLOR,
            font=('Arial', SIZES[window_size]['text_size'], 'bold')
        )
        # - localization
        self.__informations_text_1.place(relx=0.01, rely=0.01, relwidth=0.978, relheight=0.1)
        # -- results
        self.__results_frame = tk.Label(master=self.__informations_frame)
        # -- text
        self.__results_frame.config(
            text=f'Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}',
            fg=TEXT_COLOR_1,
            background=BACKGROUND_COLOR,
            font=('Arial', SIZES[window_size]['text_size'], 'bold')
        )
        # -- localization
        self.__results_frame.place(relx=0.01, rely=0.2, relwidth=0.978, relheight=0.3)

    def next_image(self) -> None:
        self.__current_index += 1
        if self.__current_index == self.__n_elements:
            self.__current_index = 0
        self.__image_frame.configure(image=self.__data[self.__current_index])
        self.__image_frame.image = self.__data[self.__current_index]

    def previous_next(self) -> None:
        self.__current_index -= 1
        if self.__current_index < 0:
            self.__current_index = self.__n_elements - 1
        self.__image_frame.configure(image=self.__data[self.__current_index])
        self.__image_frame.image = self.__data[self.__current_index]
        


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', '-ws', type=int, default=2, help='Specify the size of window.')
    parser.add_argument('--version', '-v', type=str, default='v1', help='Specify the model version to be used.')
    parser.add_argument('--cpu', '-c', action='store_true', help='Set if you want to run on CPU.')
    arg = parser.parse_args()
    
    Screen(arg.window_size, arg.version, arg.cpu).mainloop()