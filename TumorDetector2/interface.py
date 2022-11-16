# imports
from TumorDetector2.utils.weights import load_architecture
from TumorDetector2.utils.metrics import overall_results
from TumorDetector2.utils.graphs import apply_masks
from TumorDetector2.utils.data import test_dataset

from tensorflow._api.v2.config import set_visible_devices
from tensorflow.keras.utils import array_to_img
from PIL import ImageTk, Image
from tqdm import tqdm

import tkinter as tk
import numpy as np


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

        # changing device
        if on_cpu:
            set_visible_devices([], 'GPU')

        # loading model and images
        __model = load_architecture(version=version)
        __images = test_dataset()
        
        self.__data = []
        self.__labels = []
        self.__pred_labels = []
        
        # inference
        for iter, (__image, __mask, __label) in enumerate(tqdm(iterable=__images, desc='Running inferences')):
            
            # converting images from tensor to pillow
            __pred = np.asmarray(array_to_img(
                        x=__model(__image)[0], data_format='channels_last', dtype='float32'
                    ).resize(SIZES[window_size]['image_size'])) * 255
            __image = np.asmarray(array_to_img(
                        x=__image[0], data_format='channels_last', dtype='float32'
                    ).resize(SIZES[window_size]['image_size']))
            __mask = np.asmarray(array_to_img(
                x=__mask[0], data_format='channels_last', dtype='float32'
            ).resize(SIZES[window_size]['image_size']))
            
            # saving labels
            self.__labels.append(__label)
            self.__pred_labels.append(1 if __pred.sum() else 0)
            
            # overlapping results
            __segmentation = apply_masks(__image, __mask, __pred)
            self.__data.append(ImageTk.PhotoImage(image=Image.fromarray(__segmentation)))
            if iter > 30:
                break
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
        # logo
        # - loading image
        __brain_image = Image.open(fp='docs/images/brain_logo.png')
        # - resizing image
        __brain_image = __brain_image.resize(size=SIZES[window_size]['icon_size'])
        # - applying image
        __brain_image = ImageTk.PhotoImage(image=__brain_image)
        __brain_frame = tk.Label(self, image=__brain_image)
        __brain_frame.image = __brain_image
        # - logo localization
        __brain_frame.place(relx=0.4, rely=0.0, relwidth=0.2, relheight=0.1)
        # - background color
        __brain_frame.config(background=BACKGROUND_COLOR)

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
        

