# imports
from tensorflow._api.v2.config import list_physical_devices

import matplotlib as mat
import tensorflow as tf
import sklearn as sk
import pandas as pd
import numpy as np
import ipykernel
import platform
import unittest
import tqdm
import PIL



# main test case
class TestEnvironment(unittest.TestCase):
    
    def test_packages_versions(self):
        """Verify all packages versions.
        """
        
        # versions
        __packeges = (
            [platform.__name__, platform.python_version(), '3.9.13'],
            [tf.__name__, tf.__version__, '2.6.0'],
            [np.__name__, np.__version__, '1.19.5'],
            [tqdm.__name__, tqdm.__version__, '4.64.1'],
            [ipykernel.__name__, ipykernel.__version__, '6.17.1'],
            [pd.__name__, pd.__version__, '1.4.4'],
            [mat.__name__, mat.__version__, '3.6.2'],
            [PIL.__name__, PIL.__version__, '9.3.0'],
            [sk.__name__, sk.__version__, '1.1.3']
        )
        
        for name, installed, required in __packeges:
            self.assertEqual(first=installed, second=required, msg=f'Wrong {name} version.')
    
    def test_gpu_availability(self):
        """Verify if GPU exists.
        """
        self.assertGreater(
            a=list_physical_devices(device_type='GPU').__len__(),
            b=0,
            msg='No GPU has been detected. To enable GPU acceleration make sure you have installed in your computer CUDA:11.2 and cuDNN:8.1 - Compatibilities: https://www.tensorflow.org/install/source_windows?hl=pt-br#gpu'
        )


if __name__=='__main__':
    unittest.main()