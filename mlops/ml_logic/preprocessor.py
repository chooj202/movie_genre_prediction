import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.preprocessing import MultiLabelBinarizer

def binarize_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binarize genre column
    """
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(df["genre"])

    # transform target variable
    y = multilabel_binarizer.transform(df["genre"])
    genre_names = multilabel_binarizer.classes_

    # Adding binary columns to main df
    for i in range(len(genre_names)):
        df[f"{genre_names[i]}"] = y[:,i]

    return df
