

preprocess:
	python3 src/data/preprocess_raw_data.py -f all

sample:
	python3 src/data/sample_processed_data.py

create_experiment:
	python3 src/data/create_experiment.py

prepare_data: preprocess sample create_experiment
	