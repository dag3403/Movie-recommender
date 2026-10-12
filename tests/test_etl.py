import unittest

import numpy as np
import polars as pl

from peliculas.etl import tfidf_transform


class TfidfTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptions = pl.DataFrame(
            {
                "description": [
                    "alpha alpha alpha beta gamma",
                    "alpha beta beta delta",
                    "alpha gamma delta",
                    "beta gamma delta",
                ]
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
        )

        self.assertTrue(all(count == 3 for count in features.getnnz(axis=1)))

    def test_rejects_nonpositive_per_description_term_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_terms_per_document"):
            tfidf_transform(self.descriptions, top_terms_per_document=0)


if __name__ == "__main__":
    unittest.main()
