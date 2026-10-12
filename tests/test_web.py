import unittest

from peliculas.web import MovieExplorer, _nonnegative_int, _positive_int


class MovieExplorerHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.explorer = MovieExplorer.__new__(MovieExplorer)
        self.explorer.catalog_rows = [
            {
                "show_id": "m1",
                "type": "Movie",
                "title": "A River",
                "release_year": 2020,
                "duration": "90 min",
                "rating": "PG",
                "country": "Canada",
                "listed_in": "Drama",
            },
            {
                "show_id": "m2",
                "type": "Movie",
                "title": "A River",
                "release_year": 2022,
                "duration": "95 min",
                "rating": "PG-13",
                "country": "United States",
                "listed_in": "Drama, Romance",
            },
            {
                "show_id": "s1",
                "type": "TV Show",
                "title": "Other Story",
                "release_year": 2019,
                "duration": "1 Season",
                "rating": "TV-14",
                "country": None,
                "listed_in": "International TV",
            },
        ]

    def test_catalog_search_is_case_insensitive_and_pages_results(self) -> None:
        result = self.explorer.catalog_page("RIVER", 1, 1, "title", "asc")
        self.assertEqual(result["catalog_total"], 3)
        self.assertEqual(result["filtered_total"], 2)
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["items"][0]["show_id"], "m1")

        next_page = self.explorer.catalog_page("river", 2, 1, "title", "asc")
        self.assertEqual(next_page["items"][0]["show_id"], "m2")

    def test_catalog_can_sort_by_year_descending(self) -> None:
        result = self.explorer.catalog_page("", 1, 10, "year", "desc")
        self.assertEqual(
            [row["release_year"] for row in result["items"]],
            [2022, 2020, 2019],
        )

    def test_suggestions_keep_duplicate_titles_distinguishable(self) -> None:
        matches = self.explorer.search_movies("a river", 10)
        self.assertEqual([row["show_id"] for row in matches], ["m1", "m2"])
        self.assertEqual([row["release_year"] for row in matches], [2020, 2022])

    def test_pagination_and_query_integer_validation(self) -> None:
        self.assertEqual(MovieExplorer._page([], 1, 20)["pages"], 1)
        with self.assertRaises(ValueError):
            MovieExplorer._page([], 0, 20)
        self.assertEqual(_nonnegative_int("0", "cluster"), 0)
        self.assertEqual(_positive_int("3", "page"), 3)
        with self.assertRaises(ValueError):
            _positive_int("0", "page")


if __name__ == "__main__":
    unittest.main()
