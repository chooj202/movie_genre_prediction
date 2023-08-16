reinstall_package:
	@pip uninstall -y movie_genre_prediction || :
	@pip install -e .

local_data_paths:
	mkdir -p ~/.lewagon/movie_genre_prediction/data/processed
	mkdir -p ~/.lewagon/movie_genre_prediction/training_outputs/params
	mkdir -p ~/.lewagon/movie_genre_prediction/training_outputs/metrics

run_preprocess:
	python -c 'from mlops.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from mlops.interface.main import train; train()'

run_pred:
	python -c 'from mlops.interface.main import pred; pred()'

run_api:
	uvicorn mlops.api.fast:app --reload
