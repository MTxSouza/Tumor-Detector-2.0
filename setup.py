# imports
from setuptools import setup,\
                       find_packages


setup(
    name='Tumor-Detector-2.0',
    author='Matheus Oliveira de Souza',
    author_email='msouza.os@hotmail.com',
    version='1.0.0',
    description='A neural net which detect and spot any tumor in brain tomography images.',
    long_description=open('README', 'r').read(),
    license='LICENSE',
    packages=find_packages(),
    python_requires='==3.9.13',
    install_requires=[
        'tensorflow==2.6.0',
        'keras==2.6.0',
        'ipykernel==6.17.1',
        'matplotlib==3.6.2',
        'pandas==1.4.4',
        'numpy==1.19.5',
        'scikit-learn==1.1.3',
        'tqdm==4.64.1',
        'Pillow==9.3.0'
    ]
)