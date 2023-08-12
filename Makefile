reinstall_package:
	@pip uninstall -y movie_genre_prediction || :
	@pip install -e .

local_data_paths:
	mkdir -p ~/.lewagon/movie_genre_prediction/data/processed
	mkdir -p ~/.lewagon/movie_genre_prediction/training_outputs/params
	mkdir -p ~/.lewagon/movie_genre_prediction/training_outputs/metrics

run_preprocess:
	python -c 'from movie_genre_prediction.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from movie_genre_prediction.interface.main import train; train()'

run_api:
	uvicorn movie_genre_prediction.api.fast:app --reload
