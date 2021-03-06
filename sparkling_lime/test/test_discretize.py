from unittest import TestCase
import numpy as np
from sparkling_lime.discretize import QuartileDiscretizer, DecileDiscretizer
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pandas as pd
from numpy.testing import assert_allclose


class TestDiscretize(TestCase):

    @classmethod
    def setUpClass(cls):
        conf = (SparkConf().setMaster("local[4]").setAppName(
            "pytest-pyspark-local-testing"))
        cls.sc = SparkContext(conf=conf)
        cls.spark = SparkSession(cls.sc)
        # Set up expensive variables to reuse
        cls.df = cls._make_data()
        cls.qd = QuartileDiscretizer(cls.df, ["f0"], cls.df.columns)
        cls.dd = DecileDiscretizer(cls.df, ["f0"], cls.df.columns)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()

    @classmethod
    def _make_data(cls, n_rows=50, arr=False):
        """
        Helper function for generating pyspark dataframes for tests.
        """
        np.random.seed(42)
        y = np.random.binomial(1, 0.5, n_rows)
        X = np.zeros((n_rows, 4))
        z = y - np.random.binomial(1, 0.1, n_rows) \
            + np.random.binomial(1,  0.1, n_rows)
        z[z == -1] = 0
        z[z == 2] = 1
        # 5 relevant features
        X[:, 0] = z
        X[:, 1] = y * np.abs(
            np.random.normal(4, 1.2, n_rows)) + np.random.normal(4, 0.1, n_rows)
        X[:, 2] = y + np.random.normal(7.2, 2.8, n_rows)
        X[:, 3] = y ** 2 + np.random.normal(3.9, 1.4, n_rows)

        if arr:
            return X
        else:
            # Combine data into pyspark dataFrame
            data = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
            df = cls.spark.createDataFrame(data)
            return df

    def test_bin_values(self):
        data_arr = self._make_data(n_rows=50, arr=True)
        num_bins = data_arr.shape[1]
        with self.subTest("Quartile Bins are approximately correct"):
            exact_bins = []
            for feature in range(1, num_bins):     # First is categorical
                qts = np.array(np.percentile(data_arr[:, feature], [25, 50, 75]))
                exact_bins.append(qts.tolist())
            approx_bins = self.qd.bins(self.df, None)
            for bins in zip(exact_bins, approx_bins):
                assert_allclose(bins[0], bins[1], atol=0.5, err_msg=
                                    "Bin values are not close to the exact "
                                    "values.")

        with self.subTest("Decile Bins are approximately correct"):
            exact_bins = []
            for feature in range(1, num_bins):     # First is categorical
                qts = np.array(np.percentile(data_arr[:, feature],
                                             list(range(10, 100, 10))))
                exact_bins.append(qts.tolist())
            approx_bins = self.dd.bins(self.df, None)
            for bins in zip(exact_bins, approx_bins):
                assert_allclose(bins[0], bins[1], atol=0.5, err_msg=
                                    "Bin values are not close to the exact "
                                    "values.")

    def test_feature_names(self):
        pass
        # TODO (viz): Feature names aren't used since there aren't viz for now

    def test_discretizer_consistency(self):
        """
        If data that has been discretized (dfA) and undiscretized (dfB)
        is discretized again (dfC), then the original discretized dataset
        should be equal to the re-discretized dataset (dfA == dfC).
        """
        self.maxDiff = None
        columns = self.df.columns
        with self.subTest("QuartileDiscretizer has consistent behavior"):
            disc1 = self.qd.discretize(self.df)
            udisc = self.qd.undiscretize(disc1)
            disc2 = self.qd.discretize(udisc)
            self.assertCountEqual(disc1.select(sorted(disc1.columns)).collect(),
                                  disc2.select(sorted(disc2.columns)).collect(),
                                  "Rows in original discretized dataset do not"
                                  " equal rows in the re-discretized dataset.")
        with self.subTest("DecileDiscretizer has consistent behavior"):
            d_disc1 = self.dd.discretize(self.df)
            d_udisc = self.dd.undiscretize(d_disc1)
            d_disc2 = self.dd.discretize(d_udisc)
            self.assertCountEqual(
                d_disc1.select(sorted(d_disc1.columns)).collect(),
                d_disc2.select(sorted(d_disc2.columns)).collect(),
                "Rows in original discretized dataset do not"
                " equal rows in the re-discretized dataset.")
