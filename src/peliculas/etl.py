import string
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.pylab import plot
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, normalize


def load_data(directory: Path) -> pl.DataFrame:
    return pl.read_csv(directory)


def filters_movies(data: pl.DataFrame) -> pl.DataFrame:
    return data.filter(pl.col("type") == "Movie")


def drop_variables(data: pl.DataFrame, var: list[str]) -> pl.DataFrame:
    return data.select(var)


def drop_null(data: pl.DataFrame) -> pl.DataFrame:
    return data.drop_nulls()


def fill_nan(data: pl.DataFrame) -> pl.DataFrame:
    for column, dtype in data.schema.items():
        if dtype in (pl.Float32, pl.Float64):
            data = data.with_columns(pl.col(column).fill_nan(None))
    return data


def get_titles(data: pl.DataFrame) -> pl.Series:
    return data.get_column("title")


def create_numeric_features(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        pl.col("release_year").cast(pl.Float64),
        pl.col("duration")
        .str.extract(r"^(\d+)\s*min$", 1)
        .cast(pl.Float64)
        .alias("duration_min"),
    )


def scale_feature(data: pl.DataFrame, column: str) -> pl.DataFrame:
    mean = data.get_column(column).mean()
    standard_deviation = data.get_column(column).std(ddof=0)
    if standard_deviation:
        return data.with_columns(
            ((pl.col(column) - mean) / standard_deviation).alias(column)
        )
    return data.with_columns(pl.lit(0.0).alias(column))


def scale_numeric_features(data: pl.DataFrame) -> pl.DataFrame:
    data = scale_feature(data, "release_year")
    data = scale_feature(data, "duration_min")
    return data


def encode_ratings(data: pl.DataFrame) -> pl.DataFrame:
    ratings = [rating.strip() for rating in data.get_column("rating")]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    values = encoder.fit_transform([[rating] for rating in ratings])
    return pl.DataFrame(
        values.tolist(),
        schema=encoder.get_feature_names_out(["rating"]).tolist(),
        orient="row",
    )


def split_categories(values: pl.Series) -> list[list[str]]:
    rows = []
    for value in values:
        categories = [category.strip() for category in value.split(",")]
        rows.append(
            list(dict.fromkeys(category for category in categories if category))
        )
    return rows


def encode_multiple_categories(values: pl.Series, prefix: str) -> pl.DataFrame:
    rows = split_categories(values)
    encoder = MultiLabelBinarizer()
    encoded_values = encoder.fit_transform(rows)
    columns = [f"{prefix}_{category}" for category in encoder.classes_]
    return pl.DataFrame(encoded_values.tolist(), schema=columns, orient="row")


def encode_countries(data: pl.DataFrame) -> pl.DataFrame:
    return encode_multiple_categories(data.get_column("country"), "country")


def encode_genres(data: pl.DataFrame) -> pl.DataFrame:
    return encode_multiple_categories(data.get_column("listed_in"), "genre")


def join_features(
    numeric: pl.DataFrame,
    ratings: pl.DataFrame,
    countries: pl.DataFrame,
    genres: pl.DataFrame,
) -> pl.DataFrame:
    return pl.concat([numeric, ratings, countries, genres], how="horizontal")


def print_final_counts(features: pl.DataFrame) -> None:
    print(f"Final feature rows: {features.height}")
    print(f"Final feature count: {features.width}")

def tfidf_transform(data: pl.DataFrame) -> tuple:
    texts = data["description"].fill_null("").to_list()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.25)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def _complete_characteristic_movies(data: pl.DataFrame) -> pl.DataFrame:
    selected = drop_variables(
        data,
        [
            "title",
            "release_year",
            "rating",
            "duration",
            "country",
            "listed_in",
            "description",
        ],
    )
    return drop_null(fill_nan(selected))


def _described_movies(data: pl.DataFrame) -> pl.DataFrame:
    return filters_movies(data).filter(pl.col("description").is_not_null())


def process_data_characteristics(data: pl.DataFrame) -> pl.DataFrame:
    print(f"Rows before filtering: {data.height}")
    movies = filters_movies(data)
    print(f"Movie rows: {movies.height}")

    complete = _complete_characteristic_movies(movies)

    print(f"Rows removed for null/NaN values: {movies.height - complete.height}")
    print(f"Complete movie rows: {complete.height}")

    titles = get_titles(complete)
    numeric = create_numeric_features(complete)
    numeric = scale_numeric_features(numeric)

    ratings = encode_ratings(complete)
    countries = encode_countries(complete)
    genres = encode_genres(complete)
    features = join_features(numeric, ratings, countries, genres)

    print_final_counts(features)
    
    return features


def process_data_description(data: pl.DataFrame) -> pl.DataFrame:
    data = _described_movies(data)
    data = drop_variables(data, ["description"])
    data = fill_nan(data)
    vectorizer, tfidf_matrix = tfidf_transform(data)
    pprint(vectorizer)
    pprint(tfidf_matrix)
    return tfidf_matrix


def n_optimo_clusters(data: pl.DataFrame) -> None:
    inercia = []
    silhouette_scores = []
    rango_k = range(2, 20)

    for k in rango_k:
        kmeans = KMeans(n_clusters = k, random_state = 42, n_init = "auto")
        kmeans.fit(data)

        inercia.append(kmeans.inertia_)

        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rango_k, inercia, marker='o', linestyle='--')
    plt.title('Método del Codo')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Inercia (Suma de errores al cuadrado)')
    plt.xlim(min(rango_k) - 0.5, max(rango_k) + 0.5)
    plt.xticks(list(rango_k))

    plt.subplot(1, 2, 2)
    plt.plot(rango_k, silhouette_scores, marker='s', color='orange', linestyle='--')
    plt.title('Coeficiente de Silueta')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

def kmeans_clustering(data: pl.DataFrame, n_clusters: int, tipo: str) -> pl.DataFrame:
    if tipo == "char":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto"
        )

    elif tipo == "desc":
        data = normalize(data)
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto"
        )
    else:
        raise ValueError("La variables deben ser 'char' o 'desc'.")

    labels = model.fit_predict(data)

    print(labels)
    return labels

def recomendador(
    movie: str,
    n: int,
    tipo: str,
    data: pl.DataFrame,
    features: pl.DataFrame | np.ndarray | spmatrix,
    labels: np.ndarray,
) -> pl.DataFrame:
    if not movie.strip():
        raise ValueError("El título de la película no puede estar vacío.")
    if n < 1:
        raise ValueError("n debe ser un número mayor que 0.")
    if tipo not in ("char", "desc"):
        raise ValueError("tipo debe ser 'char' o 'desc'.")

    if tipo == "char":
        titles = get_titles(
            _complete_characteristic_movies(filters_movies(data))
        ).to_list()
        distance_metric = "euclidean"
    else:
        titles = get_titles(_described_movies(data)).to_list()
        distance_metric = "cosine"

    if len(titles) != features.shape[0] or len(labels) != features.shape[0]:
        raise ValueError(
            "Los títulos, las características y las etiquetas deben estar alineados."
        )

    matching_indices = [
        index
        for index, title in enumerate(titles)
        if title is not None and title.strip().casefold() == movie.strip().casefold()
    ]
    if not matching_indices:
        raise ValueError(f"No se encontró la película '{movie}'.")

    target_index = matching_indices[0]
    distances = pairwise_distances(
        features[target_index : target_index + 1],
        features,
        metric=distance_metric,
    )[0]

    candidates = [
        index
        for index, label in enumerate(labels)
        if label == labels[target_index] and index != target_index
    ]
    candidates.sort(key=lambda index: distances[index])
    recommendations = candidates[:n]

    return pl.DataFrame(
        {
            "title": [titles[index] for index in recommendations],
            "distance": [float(distances[index]) for index in recommendations],
        }
    )
