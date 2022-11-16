env-create:
	conda create --name=tumor2 python=3.9.13
	echo 'A conda environment has been created called: tumor2.'
	echo 'Activation command: $ conda activate tumor2'

env-remove:
	conda env remove --name=tumor2
	echo 'tumor2 environment has been removed.'