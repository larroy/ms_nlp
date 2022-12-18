import seaborn as sns
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from langdetect import detect
from langdetect import LangDetectException

def class_balance(df: pd.DataFrame) -> None:
    print("Class balance:")
    sns.histplot(data=df, x="Label")
    plt.show()
    labels = np.unique(df['Label']).tolist() 
    for label in labels:
        count = np.count_nonzero(df['Label']==label)
        print(f"Label: '{label}' count: {count}")


# Handle errors detecting language
def detect_language(x: str):
    try:
        return detect(x)
    except LangDetectException as e:
        #logging.exception("Error detecting language for text: '%s'", x)
        return 'unknown'

def language(df: pd.DataFrame, col: str) -> None:
    print(f"Langage for column: '{col}' ")
    lang = df[col].apply(detect_language)
    lang.hist()
    plt.show()
    df[col][lang[lang=='unknown'].index[0:10]]


def run_eda(df: pd.DataFrame) -> None:
    class_balance(df)
    #plt.ioff()
    plt.figure()
    language(df, 'Subject')
    plt.figure()
    language(df, 'Body')


