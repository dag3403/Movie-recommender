import unittest

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from peliculas.etl import recomendador, tfidf_transform


class TfidfTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptions = pl.DataFrame(
            {
                "description": [
                    "alpha alpha alpha beta gamma",
                    "alpha beta beta delta",
                    "alpha gamma delta",
                    "beta gamma delta",
                ],
                "title": ["zephyr title", "second title", "third title", "fourth title"],
            }
        )

    def test_default_keeps_at_most_fifteen_terms_per_description(self) -> None:
        _, features = tfidf_transform(self.descriptions)

        self.assertEqual(features.shape[0], self.descriptions.height)
        self.assertTrue(all(count <= 15 for count in features.getnnz(axis=1)))
        np.testing.assert_allclose(
            features.power(2).sum(axis=1).A1 ** 0.5,
            1.0,
        )

    def test_top_term_is_selected_for_each_description(self) -> None:
        vectorizer, features = tfidf_transform(
            self.descriptions,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
            sublinear_tf=False,
            top_terms_per_document=1,
            include_title=False,
        )
        vocabulary = vectorizer.get_feature_names_out()

        self.assertEqual(features.getnnz(axis=1).tolist(), [1, 1, 1, 1])
        first_term = vocabulary[features[0].indices[0]]
        self.assertEqual(first_term, "alpha")

    def test_none_disables_per_description_term_limit(self) -> None:
        _, features = tfidf_transform(
            self.descriptions,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
            top_terms_per_document=None,
            include_title=False,
        )

        self.assertTrue(all(count == 3 for count in features.getnnz(axis=1)))

    def test_rejects_nonpositive_per_description_term_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_terms_per_document"):
            tfidf_transform(self.descriptions, top_terms_per_document=0)

    def test_title_terms_are_added_to_the_description_vector(self) -> None:
        vectorizer, features = tfidf_transform(
            self.descriptions,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
            top_terms_per_document=None,
        )

        vocabulary = vectorizer.get_feature_names_out()
        self.assertIn("zephyr", vocabulary)
        title_term = vectorizer.vocabulary_["zephyr"]
        self.assertGreater(features[0, title_term], 0)


class RecommenderTests(unittest.TestCase):
    def test_recommends_nearest_movies_without_cluster_labels(self) -> None:
        catalog = pl.DataFrame(
            {
                "show_id": ["source", "nearest", "farthest"],
                "type": ["Movie", "Movie", "Movie"],
                "title": ["Source", "Nearest", "Farthest"],
                "description": ["source story", "near story", "far story"],
            }
        )
        features = csr_matrix([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])

        recommendations = recomendador(
            movie="Source",
            n=2,
            tipo="desc",
            data=catalog,
            features=features,
            reference_show_id="source",
            include_show_ids=True,
        )

        self.assertEqual(
            recommendations.get_column("show_id").to_list(),
            ["nearest", "farthest"],
        )
        self.assertEqual(recommendations.get_column("title").to_list(), ["Nearest", "Farthest"])
        self.assertNotIn("source", recommendations.get_column("show_id").to_list())
        self.assertTrue(
            recommendations.get_column("distance").is_sorted()
        )


if __name__ == "__main__":
    unittest.main()
