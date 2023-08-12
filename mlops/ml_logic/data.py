import pandas as pd

from colorama import Fore, Style
from pathlib import Path
from google.cloud import bigquery

from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
import ast

from movie_genre_prediction.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW
    # df = df.astype(DTYPES_RAW)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.drop_duplicates().reset_index(drop=True)
    df["genre"] = (
        df["genre"]
        .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        .apply(lambda x: [genre.strip().lower().replace("-", "_") for genre in x])
        )
    print("✅ data cleaned")

    return df

def image_data(df: pd.DataFrame) -> pd.DataFrame:
    # resize all images to 350x350
    # convert images to arrays (ignore corrupted images)
    width, height = 350, 350
    image_array = []
    unidentified_count = 0
    not_found_count = 0
    for i in tqdm(range(df.shape[0])):
        try:
            image_path = f"raw_data/posters/all/{df['imdb_id'][i]}.jpg"
            img = image.load_img(image_path, target_size=(width, height, 3))
            input_arr = image.img_to_array(img)
            input_arr = input_arr/255.0
            image_array.append([df['imdb_id'][i], input_arr])
        except UnidentifiedImageError as e1:
            unidentified_count += 1
            pass
        except FileNotFoundError as e2:
            not_found_count += 1
            pass
    print(f"{unidentified_count} files were unidentified and {not_found_count} files were not found")
    img_array_df = pd.DataFrame(image_array, columns=["imdb_id", "image_array"])

    # join image_array_df and df on imdb_id (to delete imdb_id with corrupted images)
    raw_genre_img_df = df.merge(img_array_df, on="imdb_id", how="right").drop(columns='image_url')
    raw_genre_img_df = raw_genre_img_df.dropna().reset_index(drop=True)
    raw_genre_img_df["image_array"] = np.array(raw_genre_img_df["image_array"])
    raw_genre_img_df["image_array"] = raw_genre_img_df["image_array"].apply(np.ravel)
    print(raw_genre_img_df["image_array"][0].shape)
    return raw_genre_img_df

def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    # if cache_path.is_file():
    if False:
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server... This might take a while..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            # for (columnName, columnData) in df.iteritems():
            #     if isinstance(columnData[0], (np.ndarray, np.generic)):
            #         df[columnName] = df[columnName].apply(lambda x: x.tolist())
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
