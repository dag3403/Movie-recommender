from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

import polars as pl

def load_data(directory: Path) -> pl.DataFrame:
    return pl.read_csv(directory)

def filters_movies(data: pl.DataFrame) -> pl.DataFrame:
    return data.filter(pl.col("type") == "Movie")

def filters_variables(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(["description"])

def tfidf_transform(data: pl.DataFrame) -> tuple:
    texts = data["description"].fill_null("").to_list()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.25)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix
    
def process_data_description(data: pl.DataFrame) -> pl.DataFrame:
    data = filters_movies(data)
    data = filters_variables(data)
    vectorizer, tfidf_matrix = tfidf_transform(data)
    pprint(vectorizer)
    pprint(tfidf_matrix)
    return data