from pathlib import Path

import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def load_data(directory: Path) -> pl.DataFrame:
    return pl.read_csv(directory)


def filters_movies(data: pl.DataFrame) -> pl.DataFrame:
    return data.filter(pl.col("type") == "Movie")


def drop_variables(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        [
            "title",
            "release_year",
            "rating",
            "duration",
            "country",
            "listed_in",
        ]
    )

def drop_variables2(data: pl.DataFrame, var: list[str]) -> pl.DataFrame:
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


def process_data_characteristics(data: pl.DataFrame) -> tuple[pl.Series, pl.DataFrame]:
    print(f"Rows before filtering: {data.height}")
    movies = filters_movies(data)
    print(f"Movie rows: {movies.height}")

    #selected = drop_variables(movies)
    selected = drop_variables2(movies, [
            "title",
            "release_year",
            "rating",
            "duration",
            "country",
            "listed_in",
        ])
    selected = fill_nan(selected)
    complete = drop_null(selected)

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
    return titles, features
