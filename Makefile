ML_DIR_DATA=/data
ML_DIR_TRAINING=/training_outputs

reinstall_package:
	@pip uninstall -y movie_genre_prediction || :
	@pip install -e .

local_data_paths:
	rm -rf ${ML_DIR_DATA}
	rm -rf ${ML_DIR_TRAINING}
	mkdir -p /data/processed
	mkdir -p /training_outputs/params
	mkdir -p /training_outputs/metrics
	mkdir -p /training_outputs/models

run_preprocess:
	python -c 'from mlops.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from mlops.interface.main import train; train()'

run_pred:
	python -c 'from mlops.interface.main import pred; pred()'

run_api:
	uvicorn mlops.api.fast:app --reload

run_local_dev_api:
	docker build --tag=${GCR_IMAGE}:dev .
	docker run -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp_credentials.json -e PORT=8000 -v ${GOOGLE_APPLICATION_CREDENTIALS}:/tmp/keys/gcp_credentials.json:ro -p 8000:8000 --env-file .env ${GCR_IMAGE}:dev


## for all other users
run_local_prod_api:
	docker build -t ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod .
	docker run -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp_credentials.json -e PORT=8000 -v ${GOOGLE_APPLICATION_CREDENTIALS}:/tmp/keys/gcp_credentials.json:ro -p 8000:8000 --env-file .env ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod

push_prod_api:
	docker push ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod


## for m1 users
run_build_intel_api_from_m1:
	docker build -t ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:intel --platform=linux/amd64 .

push_intel_api_from_m1:
	docker push ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:intel

deploy_intel_api_from_m1:
	gcloud run deploy --image ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:intel --memory ${GCR_MEMORY} --region ${GCP_REGION} --env-vars-file .env.yaml
