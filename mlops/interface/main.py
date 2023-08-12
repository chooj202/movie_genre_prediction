import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from sklearn.model_selection import train_test_split

from movie_genre_prediction.ml_logic.data import get_data_with_cache, clean_data, image_data, load_data_to_bq
from movie_genre_prediction.ml_logic.preprocessor import binarize_genres
from movie_genre_prediction.ml_logic.registry import load_model, save_results, save_model
from movie_genre_prediction.ml_logic.model import initialize_model, compile_model, train_model
from movie_genre_prediction.params import *

def preprocess() -> pd.DataFrame:
    """
    - Query the raw dataset from raw_data/{your path}
    - Process query data
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    path_ls = [f'raw_data/500_points/{genre_name}.csv' for genre_name in GENRE_NAMES]
    df = pd.concat(map(pd.read_csv, path_ls), ignore_index=True)

    # Process data
    print(Fore.MAGENTA +"\nCleaning data"+ Style.RESET_ALL)
    data_clean = clean_data(df)
    print(Fore.MAGENTA +"\nTaking image_data as vectors"+ Style.RESET_ALL)
    data_with_img = image_data(data_clean)
    print(Fore.MAGENTA +"\nBinarizing the genres"+ Style.RESET_ALL)
    data_processed = binarize_genres(data_with_img).head(int(DATA_SIZE))
    # TODO: add cache function
    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    data_processed_to_send = data_processed.drop(columns=["genre"])
    print(data_processed_to_send.head(1))

    load_data_to_bq(
        data_processed_to_send,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_{MODEL_NAME}_{DATA_SIZE}',
        truncate=True
    )

    print("✅ Preprocess() done \n")
    print(f"A row from the df\n {data_processed.head(1)}")


def train(
        split_ratio: float = 0.1,
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order

    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{MODEL_NAME}_{DATA_SIZE}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{MODEL_NAME}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )
    print(Fore.BLUE + "First row of data:" + Style.RESET_ALL + f"\n {data_processed.head(1)}")

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    data_processed["image_array"] = (data_processed["image_array"]
        .apply(lambda x: np.reshape(x, (350, 350, 3)))
        )
    X = data_processed["image_array"].values
    y = data_processed.drop(columns=["movie", "imdb_id","plot", "image_array"], axis=1)
    X = np.array(list(X))
    y = y.to_numpy().astype('int64')
    X_train_processed,X_val_processed,y_train,y_val=train_test_split(X,y,test_size=split_ratio)

    print(Fore.GREEN + f"X_train[0] shape: \n{X_train_processed[0].shape}"+ Style.RESET_ALL)
    print(Fore.GREEN + f"y_train[0] shape:\n{y_train[0].shape}" + Style.RESET_ALL)

    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model(input_shape=X_train_processed[0].shape, output_shape=y.shape[1])

    model = compile_model(model)
    model, history = train_model(
        model, X_train_processed, y_train, X_val_processed, y_val
    )

    accuracy = history.history['accuracy'][-1]

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive
    save_results(params=params, metrics=dict(accuracy=accuracy))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return accuracy
