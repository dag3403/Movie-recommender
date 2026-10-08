from pprint import pprint

from peliculas import consts
from peliculas.etl import (
    kmeans_clustering,
    load_data,
    n_optimo_clusters,
    process_data_characteristics,
    process_data_description,
    recomendador,
)


def main() -> None:
    data = load_data(consts.DATA_PATH / "netflix_titles.csv")
    data_char = process_data_characteristics(data)
    data_desc = process_data_description(data)
    #n_optimo_clusters(data_char)
    #n_optimo_clusters(data_desc)
    # One-Hot / Multi-Hot
    labels_char = kmeans_clustering(data_char, 2, "char")
    labels_desc = kmeans_clustering(data_desc, 15, "desc")

    movie = input("Introduce el título de una película: ").strip()
    n = int(input("¿Cuántas películas similares quieres recibir?: "))
    tipo = input("¿Qué tipo de recomendación quieres? ('char' o 'desc'): ").strip()

    if tipo == "char":
        features = data_char
        labels = labels_char
    elif tipo == "desc":
        features = data_desc
        labels = labels_desc
    else:
        raise ValueError("El tipo debe ser 'char' o 'desc'.")

    recommendations = recomendador(movie, n, tipo, data, features, labels)
    print(recommendations)