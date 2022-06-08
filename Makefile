.PHONY= all clean clean_experiments

PYTHON=python3
DATA=src/data
EXPERIMENT=src/experiment
DATACONFIG=config/data
MODELCONFIG=config/model
EXPERIMENTCONFIG=config/experiment
EPOCHS=200

preprocess: src/data/preprocess_raw_data.py
	${PYTHON} ${DATA}/preprocess_raw_data.py -f all

sample: src/data/sample_processed_data.py
	${PYTHON} ${DATA}/sample_processed_data.py

prepare_experiments: src/data/create_experiment_dataset.py  
	${PYTHON} ${DATA}/create_experiment_dataset.py -f ${DATACONFIG}/setup_diagonal_pressure_tensor_dataset.yaml
	${PYTHON} ${DATA}/create_experiment_dataset.py -f ${DATACONFIG}/setup_offdiagonal_pressure_tensor_dataset.yaml
	${PYTHON} ${DATA}/create_experiment_dataset.py -f ${DATACONFIG}/setup_heatflux_dataset.yaml
	
experiments: preprocess sample prepare_experiments

linreg_experiment:
	${PYTHON} ${EXPERIMENT}/train_and_predict_LR.py -c ${EXPERIMENTCONFIG}/LR/diagonal_pressure_experiment.yaml
	${PYTHON} ${EXPERIMENT}/train_and_predict_LR.py -c ${EXPERIMENTCONFIG}/LR/offdiagonal_pressure_experiment.yaml
	${PYTHON} ${EXPERIMENT}/train_and_predict_LR.py -c ${EXPERIMENTCONFIG}/LR/heatflux_experiment.yaml

hgbr_experiment:
	${PYTHON} ${EXPERIMENT}/train_and_predict_HGBR.py -c ${EXPERIMENTCONFIG}/HGBR/diagonal_pressure_experiment.yaml
	${PYTHON} ${EXPERIMENT}/train_and_predict_HGBR.py -c ${EXPERIMENTCONFIG}/HGBR/offdiagonal_pressure_experiment.yaml
	${PYTHON} ${EXPERIMENT}/train_and_predict_HGBR.py -c ${EXPERIMENTCONFIG}/HGBR/heatflux_experiment.yaml

mlp_experiment:
	${PYTHON} ${EXPERIMENT}/train_and_predict_MLP.py -c ${EXPERIMENTCONFIG}/MLP/diagonal_pressure_experiment.yaml -e ${EPOCHS}
	${PYTHON} ${EXPERIMENT}/train_and_predict_MLP.py -c ${EXPERIMENTCONFIG}/MLP/offdiagonal_pressure_experiment.yaml -e ${EPOCHS}
	${PYTHON} ${EXPERIMENT}/train_and_predict_MLP.py -c ${EXPERIMENTCONFIG}/MLP/heatflux_experiment.yaml -e ${EPOCHS}

clean_data_experiments:
	rm -rf data/experiment/*
	touch data/experiment/.gitkeep

clean_experiments:
	rm -rf experiment/*
	touch experiment/.gitkeep

clean: clean_data_experiments
	rm -rf data/processed/*.h5
	rm -rf data/sampled/*.h5
	