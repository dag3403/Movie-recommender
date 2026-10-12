from __future__ import annotations

import argparse
import json
import logging
import threading
import webbrowser
from collections import Counter, OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from peliculas import consts, etl

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
APPROACHES = {
    "characteristics": ("char", "Características"),
    "description": ("desc", "Descripción"),
}
RECOMMENDATION_CACHE_SIZE = 32


class MovieExplorer:
    def __init__(self, data_path: Path) -> None:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.catalog = etl.load_data(data_path)
            self.characteristics = etl.process_data_characteristics(self.catalog)
            self.descriptions = etl.process_data_description(self.catalog)
            self.labels = {
                "characteristics": etl.kmeans_clustering(
                    self.characteristics, 2, "char"
                ),
                "description": etl.kmeans_clustering(
                    self.descriptions, 15, "desc"
                ),
            }
            self.aligned_movies = {
                "characteristics": etl._complete_characteristic_movies(
                    etl.filters_movies(self.catalog)
                ),
                "description": etl._described_movies(self.catalog),
            }

        self.catalog_rows = self.catalog.to_dicts()
        self.movies_by_id = {
            str(row["show_id"]): row
            for row in self.catalog_rows
            if row.get("show_id") is not None
        }
        self.aligned_indices = {
            approach: {
                str(show_id): index
                for index, show_id in enumerate(
                    movies.get_column("show_id").to_list()
                )
            }
            for approach, movies in self.aligned_movies.items()
        }
        self.recommendation_cache: OrderedDict[
            tuple[str, str], list[dict[str, str | float | None]]
        ] = OrderedDict()
        self.cache_lock = threading.Lock()

    @staticmethod
    def _approach(value: str | None) -> str:
        if value not in APPROACHES:
            raise ValueError("El enfoque debe ser 'characteristics' o 'description'.")
        return value

    @staticmethod
    def _movie_summary(row: dict) -> dict:
        return {
            "show_id": row.get("show_id"),
            "type": row.get("type"),
            "title": row.get("title"),
            "release_year": row.get("release_year"),
            "duration": row.get("duration"),
            "rating": row.get("rating"),
            "country": row.get("country"),
            "listed_in": row.get("listed_in"),
        }

    def summary(self) -> dict:
        return {
            "catalog_total": len(self.catalog_rows),
            "movie_total": sum(row.get("type") == "Movie" for row in self.catalog_rows),
            "clustered": {
                approach: len(labels)
                for approach, labels in self.labels.items()
            },
            "cluster_totals": {
                approach: len(set(labels.tolist()))
                for approach, labels in self.labels.items()
            },
        }

    def cluster_summary(self, approach: str) -> dict:
        approach = self._approach(approach)
        labels = self.labels[approach]
        sizes = Counter(int(label) for label in labels)
        total = len(labels)
        clusters = [
            {
                "id": cluster_id,
                "count": count,
                "percentage": round(100 * count / total, 2),
            }
            for cluster_id, count in sorted(sizes.items())
        ]
        return {
            "approach": approach,
            "label": APPROACHES[approach][1],
            "total_clusters": len(clusters),
            "clustered_movies": total,
            "clusters": clusters,
        }

    def cluster_movies(
        self,
        approach: str,
        cluster_id: int,
        query: str,
        page: int,
        page_size: int,
    ) -> dict:
        approach = self._approach(approach)
        clusters = self.cluster_summary(approach)["clusters"]
        if cluster_id not in {cluster["id"] for cluster in clusters}:
            raise ValueError("El identificador de cluster no existe.")

        query = query.casefold().strip()
        aligned = self.aligned_movies[approach]
        labels = self.labels[approach]
        ids = aligned.get_column("show_id").to_list()
        rows = [
            self._movie_summary(self.movies_by_id[str(show_id)])
            for index, show_id in enumerate(ids)
            if int(labels[index]) == cluster_id
            and query
            in str(self.movies_by_id[str(show_id)].get("title") or "").casefold()
        ]
        return self._page(rows, page, page_size)

    def catalog_page(
        self,
        query: str,
        page: int,
        page_size: int,
        sort: str,
        direction: str,
    ) -> dict:
        if sort not in ("title", "year"):
            raise ValueError("La ordenación debe ser por título o año.")
        if direction not in ("asc", "desc"):
            raise ValueError("La dirección debe ser 'asc' o 'desc'.")

        query = query.casefold().strip()
        rows = [
            self._movie_summary(row)
            for row in self.catalog_rows
            if query in str(row.get("title") or "").casefold()
        ]
        reverse = direction == "desc"
        if sort == "title":
            rows.sort(key=lambda row: str(row.get("title") or "").casefold(), reverse=reverse)
        else:
            known_years = [row for row in rows if row.get("release_year") is not None]
            missing_years = [row for row in rows if row.get("release_year") is None]
            known_years.sort(
                key=lambda row: (
                    row["release_year"],
                    str(row.get("title") or "").casefold(),
                ),
                reverse=reverse,
            )
            rows = known_years + missing_years

        result = self._page(rows, page, page_size)
        result.update(
            {
                "catalog_total": len(self.catalog_rows),
                "filtered_total": len(rows),
            }
        )
        return result

    def search_movies(self, query: str, limit: int) -> list[dict]:
        query = query.casefold().strip()
        if not query:
            return []
        rows = [
            self._movie_summary(row)
            for row in self.catalog_rows
            if row.get("type") == "Movie"
            and query in str(row.get("title") or "").casefold()
        ]
        rows.sort(
            key=lambda row: (
                str(row.get("title") or "").casefold(),
                row.get("release_year") or 0,
                str(row.get("show_id") or ""),
            )
        )
        return rows[:limit]

    @staticmethod
    def _page(rows: list[dict], page: int, page_size: int) -> dict:
        if page < 1:
            raise ValueError("La página debe ser un número mayor que 0.")
        if not 1 <= page_size <= 100:
            raise ValueError("El tamaño de página debe estar entre 1 y 100.")
        page_count = max(1, (len(rows) + page_size - 1) // page_size)
        page = min(page, page_count)
        start = (page - 1) * page_size
        return {
            "items": rows[start : start + page_size],
            "page": page,
            "page_size": page_size,
            "pages": page_count,
            "results_count": len(rows),
        }

    def recommend(self, show_id: str, approach: str, requested: int) -> dict:
        approach = self._approach(approach)
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
            raise ValueError("El número de recomendaciones debe ser un entero mayor que 0.")
        reference = self.movies_by_id.get(show_id)
        if reference is None or reference.get("type") != "Movie":
            raise ValueError("No se encontró la película seleccionada en el catálogo.")

        target_index = self.aligned_indices[approach].get(show_id)
        if target_index is None:
            label = APPROACHES[approach][1]
            raise ValueError(
                f"'{reference.get('title')}' no tiene datos completos para el enfoque de {label.lower()}."
            )

        labels = self.labels[approach]
        target_cluster = int(labels[target_index])
        available = int(np.count_nonzero(labels == target_cluster)) - 1
        count = min(requested, max(available, 0))
        if count == 0:
            return {
                "reference": self._movie_summary(reference),
                "approach": approach,
                "requested": requested,
                "available": 0,
                "recommendations": [],
                "message": "No hay recomendaciones disponibles en el cluster de esta película.",
            }

        key = (approach, show_id)
        with self.cache_lock:
            ranked = self.recommendation_cache.get(key)
            if ranked is None:
                title = str(reference.get("title") or "")
                all_in_cluster = etl.recomendador(
                    title,
                    available,
                    APPROACHES[approach][0],
                    self.catalog,
                    self.characteristics
                    if approach == "characteristics"
                    else self.descriptions,
                    labels,
                    reference_show_id=show_id,
                    include_show_ids=True,
                )
                ranked = [
                    {
                        **self._movie_summary(self.movies_by_id[str(row["show_id"])]),
                        "distance": float(row["distance"]),
                    }
                    for row in all_in_cluster.to_dicts()
                ]
                self.recommendation_cache[key] = ranked
                self.recommendation_cache.move_to_end(key)
                if len(self.recommendation_cache) > RECOMMENDATION_CACHE_SIZE:
                    self.recommendation_cache.popitem(last=False)
            else:
                self.recommendation_cache.move_to_end(key)

        return {
            "reference": self._movie_summary(reference),
            "approach": approach,
            "requested": requested,
            "available": available,
            "recommendations": ranked[:count],
            "message": (
                f"Se muestran {count} de {requested}; el cluster ofrece "
                f"{available} recomendaciones."
                if count < requested
                else None
            ),
        }


def _query_value(query: dict[str, list[str]], name: str, default: str) -> str:
    return query.get(name, [default])[0]


def _positive_int(value: str, name: str, maximum: int = 100) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"'{name}' debe ser un número entero.") from error
    if not 1 <= result <= maximum:
        raise ValueError(f"'{name}' debe estar entre 1 y {maximum}.")
    return result


def _nonnegative_int(value: str, name: str, maximum: int = 1000) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"'{name}' debe ser un número entero.") from error
    if not 0 <= result <= maximum:
        raise ValueError(f"'{name}' debe estar entre 0 y {maximum}.")
    return result


def make_handler(explorer: MovieExplorer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict | list, content_type: str) -> None:
            body = (
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                if content_type == "application/json; charset=utf-8"
                else payload
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict | list) -> None:
            self._send(status, payload, "application/json; charset=utf-8")

        def _error(self, error: ValueError, status: int = 400) -> None:
            self._json(status, {"error": str(error)})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/summary":
                    self._json(200, explorer.summary())
                elif parsed.path == "/api/clusters":
                    approach = explorer._approach(
                        _query_value(query, "approach", "characteristics")
                    )
                    self._json(200, explorer.cluster_summary(approach))
                elif parsed.path == "/api/cluster-movies":
                    approach = explorer._approach(
                        _query_value(query, "approach", "characteristics")
                    )
                    cluster_id = _nonnegative_int(
                        _query_value(query, "cluster", "0"), "cluster"
                    )
                    self._json(
                        200,
                        explorer.cluster_movies(
                            approach,
                            cluster_id,
                            _query_value(query, "q", ""),
                            _positive_int(_query_value(query, "page", "1"), "page"),
                            _positive_int(
                                _query_value(query, "page_size", "20"),
                                "page_size",
                                100,
                            ),
                        ),
                    )
                elif parsed.path == "/api/catalog":
                    self._json(
                        200,
                        explorer.catalog_page(
                            _query_value(query, "q", ""),
                            _positive_int(_query_value(query, "page", "1"), "page"),
                            _positive_int(
                                _query_value(query, "page_size", "20"),
                                "page_size",
                                100,
                            ),
                            _query_value(query, "sort", "title"),
                            _query_value(query, "direction", "asc"),
                        ),
                    )
                elif parsed.path == "/api/search":
                    self._json(
                        200,
                        explorer.search_movies(
                            _query_value(query, "q", ""),
                            _positive_int(
                                _query_value(query, "limit", "10"), "limit", 20
                            ),
                        ),
                    )
                elif parsed.path in ("/", "/index.html", "/app.js", "/styles.css"):
                    filename = "index.html" if parsed.path in ("/", "/index.html") else parsed.path[1:]
                    path = STATIC_ROOT / filename
                    content_type = (
                        "text/html; charset=utf-8"
                        if filename.endswith(".html")
                        else "text/javascript; charset=utf-8"
                        if filename.endswith(".js")
                        else "text/css; charset=utf-8"
                    )
                    self._send(200, path.read_bytes(), content_type)
                else:
                    self._json(404, {"error": "Ruta no encontrada."})
            except ValueError as error:
                self._error(error)
            except Exception:
                LOGGER.exception("Error procesando GET %s", parsed.path)
                self._json(500, {"error": "No se pudo completar la solicitud."})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/recommend":
                self._json(404, {"error": "Ruta no encontrada."})
                return
            try:
                content_length = _positive_int(
                    self.headers.get("Content-Length", "0"),
                    "Content-Length",
                    4096,
                )
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    self._error(ValueError("La solicitud debe contener un objeto JSON."))
                    return
                show_id = payload.get("show_id")
                approach = payload.get("approach")
                requested = payload.get("count")
                if not isinstance(show_id, str) or not show_id.strip():
                    raise ValueError("Selecciona una película del catálogo.")
                if not isinstance(approach, str):
                    self._error(ValueError("Selecciona un enfoque de recomendación."))
                    return
                self._json(
                    200,
                    explorer.recommend(show_id, approach, requested),
                )
            except (ValueError, json.JSONDecodeError) as error:
                self._error(ValueError(str(error)))
            except Exception:
                LOGGER.exception("Error procesando POST /api/recommend")
                self._json(500, {"error": "No se pudieron obtener recomendaciones."})

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Explorador web de películas Netflix")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    data_path = PROJECT_ROOT / consts.DATA_PATH / "netflix_titles.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {data_path}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Cargando catálogo y calculando clusters una vez al iniciar.")
    explorer = MovieExplorer(data_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(explorer))
    url = f"http://{args.host}:{args.port}"
    LOGGER.info("Aplicación disponible en %s", url)
    threading.Timer(1, webbrowser.open_new_tab, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Cerrando la aplicación.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
