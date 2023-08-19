FROM python:3.10.6
COPY mlops /mlops
COPY requirements_prod.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY Makefile Makefile
RUN make local_data_paths
CMD uvicorn mlops.api.fast:app --host 0.0.0.0 --port $PORT
