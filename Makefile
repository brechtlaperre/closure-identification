.PHONY= all clean

PYTHON=python3
DATA=src/data

preprocess: src/data/preprocess_raw_data.py
	${PYTHON} ${DATA}/preprocess_raw_data.py -f all

sample: src/data/sample_processed_data.py
	${PYTHON} ${DATA}/sample_processed_data.py

prepare_experiments: src/data/create_experiment.py  
	${PYTHON} ${DATA}/create_experiment.py -f config/setup_diagonal_pressure_tensor_experiment.yaml
	${PYTHON} ${DATA}/create_experiment.py -f config/setup_offdiagonal_pressure_tensor_experiment.yaml
	${PYTHON} ${DATA}/create_experiment.py -f config/setup_heatflux_experiment.yaml
	
experiments: preprocess sample prepare_experiments

clean:
	rm -rf data/processed/*.h5
	rm -rf data/sampled/*.h5
	
	rm -rf data/experiment/*
	touch data/experiment/.gitkeep