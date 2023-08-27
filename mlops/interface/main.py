import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from sklearn.model_selection import train_test_split

from mlops.ml_logic.data import (
    image_preprocessing,
    get_data_with_cache,
    clean_data,
    image_data,
    load_data_to_bq,
    get_image_array_multimodal,
    preprocess_genre_multimodal,
    tokenize_encode_multimodal
)
from mlops.ml_logic.preprocessor import binarize_genres
from mlops.ml_logic.registry import load_model, save_results, save_model
from mlops.ml_logic.model import (
    initialize_model,
    compile_model,
    train_model,
    initialize_compile_multimodal,
    fit_multimodal
    )
from mlops.params import *

from tensorflow import keras

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


def pred(file_path: str = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    Takes in the path of the image as an input (because we will download all the images sent to the API)
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)
    if file_path is None:
        print(Fore.BLUE + "\nWoops, you didn't input a file path... Let me use a default file path" + Style.RESET_ALL)
        file_path = f"{SAVEIMAGEDIR}sample.jpg"

    print(Fore.BLUE + "\nProcessing image. . ." + Style.RESET_ALL)
    X_pred = image_preprocessing(file_path).reshape(1, int(IMAGE_WIDTH), int(IMAGE_HEIGHT),3)

    print(Fore.BLUE + "\nDone processing! Now loading model!" + Style.RESET_ALL)
    model = load_model()
    assert model is not None

    print(Fore.BLUE + "\nPredicting . . . " + Style.RESET_ALL)
    y_pred = model.predict(X_pred)
    classes = np.array(GENRE_NAMES)
    top3 = np.argsort(y_pred[0])[:-4:-1]
    prediction = classes[top3[0:4]]

    print(f"\n✅ prediction done for {file_path}:")
    print(Fore.BLUE + f"\nThe model predicts {prediction}" +  Style.RESET_ALL)

    return prediction


def fast_pred(model: keras.Model, file_path: str) -> np.ndarray:
    """
    Pred but takes a model as input (We could pre-load the model and just call fast_prec)
    Takes in the path of the image as an input (because we will download all the images sent to the API)
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: fast predict!!!" + Style.RESET_ALL)
    if file_path is None:
        print(Fore.BLUE + "\nWoops, you didn't input a file path... Let me use a default file path" + Style.RESET_ALL)
        file_path = f"{SAVEIMAGEDIR}sample.jpg"

    print(Fore.BLUE + "\nProcessing image. . ." + Style.RESET_ALL)
    X_pred = image_preprocessing(file_path).reshape(1, int(IMAGE_WIDTH), int(IMAGE_HEIGHT),3)
    assert model is not None

    print(Fore.BLUE + "\nPredicting . . . " + Style.RESET_ALL)
    y_pred = model.predict(X_pred)
    classes = np.array(GENRE_NAMES)
    top3 = np.argsort(y_pred[0])[:-4:-1]
    prediction = classes[top3[0:4]]

    print(f"\n✅ prediction done for {file_path}:")
    print(Fore.BLUE + f"\nThe model predicts {prediction}" +  Style.RESET_ALL)

    return prediction


def preprocess_multimodal() -> (np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Preprocess raw data from local data for multimodal
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: Preprocess for multmodal" + Style.RESET_ALL)

    big_train_df = pd.read_csv('raw_data/large_dataset/big_data_train.csv').drop(columns = "Unnamed: 0").head(4000)
    big_test_df = pd.read_csv('raw_data/large_dataset/big_data_test.csv').drop(columns = "Unnamed: 0").head(1000)
    big_val_df = pd.read_csv('raw_data/large_dataset/big_data_val.csv').drop(columns = "Unnamed: 0").head(1000)

    df_train, X_train_img = get_image_array_multimodal(big_train_df)
    df_test, X_test_img = get_image_array_multimodal(big_test_df)
    df_val, X_val_img = get_image_array_multimodal(big_val_df)

    print(Fore.CYAN + "\nFinished getting all image arrays from local" + Style.RESET_ALL)

    df_train, y_train = preprocess_genre_multimodal(df_train)
    df_test, y_test = preprocess_genre_multimodal(df_test)
    df_val, y_val = preprocess_genre_multimodal(df_val)

    print(Fore.CYAN + "\nPreprocessed genres!" + Style.RESET_ALL)

    train_encodings = tokenize_encode_multimodal(df_train)
    test_encodings = tokenize_encode_multimodal(df_test)
    val_encodings = tokenize_encode_multimodal(df_val)

    print(Fore.CYAN + "\nDone tokenizing and encoding plots" + Style.RESET_ALL)
    print(len(X_train_img), len(X_test_img), len(X_val_img))
    # X_train_img = np.array(list(df_train["image_array"].values))
    X_train_text = train_encodings['input_ids']
    # y_train = y_train
    # X_test_img = np.array(list(df_test["image_array"].values))
    X_test_text = test_encodings['input_ids']
    # y_test = y_test
    # X_val_img = np.array(list(df_val["image_array"].values))
    X_val_text = val_encodings['input_ids']
    # y_val = y_val

    print(Fore.CYAN + "\nFinsihed preparing train, test and val datasets" + Style.RESET_ALL)

    return X_train_img, X_train_text, y_train, X_test_img, X_test_text, y_test, X_val_img, X_val_text, y_val


def compile_and_train_multimodal(
    X_train_img,
    X_train_text,
    y_train,
    X_test_img,
    X_test_text,
    y_test,
    X_val_img,
    X_val_text,
    y_val
    ):

    print(Fore.MAGENTA + "\n ⭐️ Use case: Compile and train multimodal" + Style.RESET_ALL)

    model = initialize_compile_multimodal(y_train.shape[1])

    print(Fore.CYAN + "\nFinsihed building and compling model!" + Style.RESET_ALL)

    model, history = fit_multimodal(model, X_train_img, X_train_text, y_train, X_val_img, X_val_text, y_val)

    print(Fore.CYAN + "\nFinsihed fitting model!" + Style.RESET_ALL)

    save_model(model)

    print(Fore.CYAN + "\nSaved model to GCS!" + Style.RESET_ALL)

    return model, history


def multimodal():

    X_train_img, X_train_text, y_train, X_test_img, X_test_text, y_test, X_val_img, X_val_text, y_val = preprocess_multimodal()

    model, history = compile_and_train_multimodal(X_train_img, X_train_text, y_train, X_test_img, X_test_text, y_test, X_val_img, X_val_text, y_val)


if __name__ == '__main__':
    preprocess()
    train()
    pred()
