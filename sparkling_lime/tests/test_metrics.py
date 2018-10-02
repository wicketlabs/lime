from sparkling_lime import metrics
from sparkling_lime.discretize import QuartileDiscretizer
from pyspark.ml.linalg import Vectors
from unittest import TestCase
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
from pyspark.ml.tests import SparkSessionTestCase


class PairwiseDistanceTests(SparkSessionTestCase):

    def test_pairwise_distance(self):
        data = [(0, Vectors.dense([-1.0, -1.0]),),
                (1, Vectors.dense([-1.0, 1.0]),),
                (2, Vectors.dense([1.0, -1.0]),),
                (3, Vectors.dense([1.0, 1.0]),)]
        df = self.spark.createDataFrame(data, ["id", "features"])
        vec = df.filter(col("id") == 0).first().features
        vec2 = df.filter(col("id") == 1).first().features

        with self.subTest("Test default parameters exist"):
            d0 = metrics.PairwiseDistance()
            self.assertCountEqual(
                d0.params, [d0.inputCol, d0.outputCol, d0.metric, d0.rowVector])
            self.assertTrue(all([~d0.isSet(p) for p in d0.params]))
            self.assertTrue(d0.hasDefault(d0.metric))
            self.assertEqual(d0.getMetric(), "euclidean")

        with self.subTest("Test parameters are set"):
            d0.setParams(inputCol="input", outputCol="output", rowVector=vec,
                         metric="euclidean")
            self.assertTrue(all([d0.isSet(p) for p in d0.params]))
            self.assertEqual(d0.getMetric(), "euclidean")
            self.assertEqual(d0.getRowVector(), vec)
            self.assertEqual(d0.getInputCol(), "input")
            self.assertEqual(d0.getOutputCol(), "output")

        with self.subTest("Test parameters are copied properly"):
            d0c = d0.copy({d0.rowVector: vec2})
            self.assertEqual(d0c.uid, d0.uid)
            self.assertCountEqual(d0c.params, d0.params)
            self.assertEqual(d0c.getRowVector(), vec2)

        with self.subTest("New instances are populated correctly and are "
                          "distinct"):
            d1 = metrics.PairwiseDistance(
                rowVector=vec, inputCol="features", outputCol="output",
                metric="euclidean")
            self.assertNotEqual(d1.uid, d0.uid)
            self.assertEqual(d1.getMetric(), "euclidean")
            self.assertEqual(d1.getRowVector(), vec)
            self.assertEqual(d1.getInputCol(), "features")
            self.assertEqual(d1.getOutputCol(), "output")

        actual = d1.transform(df).select("id", "output").collect()
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 2.8284271247461903}
        for r in actual:
            with self.subTest(
                    "Distances are calculated correctly: i={}".format(r["id"])):
                self.assertAlmostEqual(r["output"], expected[r["id"]])


class KernelWidthTests(SparkSessionTestCase):

    def test_kernel_weight(self):
        data = [[0, 0.0], [1, 2.0], [2, 2.0], [3, 2.8284271247461903]]
        df = self.spark.createDataFrame(data, ["id", "distances"])
        with self.subTest("Test default parameters exist"):
            k0 = metrics.KernelWeight()
            self.assertCountEqual(
                k0.params, [k0.inputCol, k0.outputCol, k0.kernelWidth])
            self.assertTrue(all([~k0.isSet(p) for p in k0.params]))
            self.assertFalse(k0.hasDefault(k0.kernelWidth))

        with self.subTest("Test parameters are set"):
            k0.setParams(inputCol="input", outputCol="output", kernelWidth=1.5)
            self.assertTrue(all([k0.isSet(p) for p in k0.params]))
            self.assertEqual(k0.getKernelWidth(), 1.5)
            self.assertEqual(k0.getInputCol(), "input")
            self.assertEqual(k0.getOutputCol(), "output")

        with self.subTest("Test parameters are copied properly"):
            k0c = k0.copy({k0.kernelWidth: 2.5})
            self.assertEqual(k0c.uid, k0.uid)
            self.assertCountEqual(k0c.params, k0.params)
            self.assertEqual(k0c.getKernelWidth(), 2.5)

        with self.subTest("New instances are populated correctly and are "
                          "distinct"):
            width = 1.0606601717798214
            k1 = metrics.KernelWeight(
                kernelWidth=width, inputCol="distances", outputCol="output")
            self.assertNotEqual(k1.uid, k0.uid)
            self.assertEqual(k1.getKernelWidth(), width)
            self.assertEqual(k1.getInputCol(), "distances")
            self.assertEqual(k1.getOutputCol(), "output")

        actual = k1.transform(df).select("id", "output").collect()
        expected = {0: 1.0, 1: 0.16901332, 2: 0.16901332, 3: 0.0285655}
        for r in actual:
            with self.subTest(
                    "Distances are calculated correctly: i={}".format(r["id"])):
                self.assertAlmostEqual(r["output"], expected[r["id"]])


class NeighborhoodGeneratorTests(SparkSessionTestCase):

    def test_random_choice_binarize(self):
        data = [[0], [0], [1], [1], [1], [1], [1], [1], [1], [1]]
        rates = {0: 0.2, 1: 0.8}
        df = self.spark.createDataFrame(data, ["c1"])
        actual = metrics.NeighborhoodGenerator._make_random_choices_and_binarize(
            df, "c1", 100, rates)
        sample_row = actual.head()
        # orig_val = sample_row["c1"]
        self.assertEqual(10, actual.count(), "Wrong count. Expected 10, got {}"
                         .format(actual.count()))
        self.assertEqual(100, len(sample_row["neighborhood"]),
                         "Wrong neighborhood size. Expected 100, got {}"
                         .format(len(sample_row["neighborhood"])))
        sample_vals = [row["inverse"] for row in sample_row["neighborhood"]]
        self.assertAlmostEqual(0.8, float(np.mean(sample_vals)),
                               delta=0.1,
                               msg=("Expected 1's to be sampled at rate of 0.8"
                                    " , got {}").format(np.mean(sample_vals)))
        orig_val = sample_row["c1"]
        with self.subTest("Binary classification is correct."):
            for row in sample_row["neighborhood"]:
                if row["inverse"] == orig_val:
                    self.assertEqual(1.0, row["binary"],
                                     ("Expected 1.0 for class ('same'), but got"
                                     " {}").format(row["binary"]))
                else:
                    self.assertEqual(0.0, row["binary"],
                                     ("Expected 1.0 for class ('diff'), but got"
                                     " {}").format(row["binary"]))

    def test_make_normals(self):
        df = self.spark.createDataFrame([[i] for i in range(10)], ["c1"])
        with self.subTest("Sample around a mean works as expected"):
            means = (-1000, 100, 1000, 10, 0)
            scales = (0, 0.1, 1, 10, 1)
            actual_cols = metrics.NeighborhoodGenerator._make_normals(
                150, 5, scales, mean=means)
            actual_mean_df = df.select([c.alias(str(i))
                                        for c, i in zip(actual_cols,
                                                        range(len(actual_cols)))])
            self.assertEqual(5, len(actual_mean_df.columns),
                             "Expected 5 columns but got {}"
                             .format(len(df.columns)))
            self.assertEqual(10, actual_mean_df.count(),
                             "Expected 10 rows but got {}"
                             .format(actual_mean_df.count()))
            sample_row = actual_mean_df.head()
            for i in range(5):
                if means[i] == 0:
                    delta = 0.2
                else:
                    delta = means[i]*0.2
                self.assertEqual(150, len(sample_row[i]),
                                 "Expected 150 samples but got {}"
                                 .format(len(sample_row[i])))
                self.assertAlmostEqual(
                    means[i], float(np.mean(sample_row[i])), delta=delta,
                    msg=("Expected mean of {} but got {}"
                         .format(means[i], np.mean(sample_row[i]))))
                self.assertAlmostEqual(
                    scales[i], float(np.std(sample_row[i])),
                    delta=scales[i]*0.2,
                    msg=("Expected scale of {} but got {}"
                         .format(scales[i], np.std(sample_row[i]))))

        with self.subTest("Sample around values works as expected"):
            cols = [col("c1")]
            scales = [1]
            actual_cols = metrics.NeighborhoodGenerator._make_normals(
                150, 1, scales, cols=cols)
            actual_sample_df = df.select([c.alias(str(i)) for c, i
                                          in zip(actual_cols,
                                                 range(len(actual_cols)))])
            self.assertEqual(1, len(actual_sample_df.columns),
                             "Expected 1 column but got {}"
                             .format(len(actual_sample_df.columns)))
            self.assertEqual(10, actual_sample_df.count(),
                             "Expected 10 rows but got {}"
                             .format(len(actual_sample_df.columns)))
            collection = actual_sample_df.collect()
            for i in range(10):
                if i == 0:
                    delta = 0.2
                else:
                    delta = i*0.2
                self.assertEqual(150, len(collection[i][0]),
                                 "Expected 150 samples but got {}"
                                 .format(len(collection[i][0])))
                self.assertAlmostEqual(i, float(np.mean(collection[i][0])),
                                       delta=delta,
                                       msg=("Expected mean {} but got {}"
                                       .format(i, np.mean(collection[i][0]))))
                self.assertAlmostEqual(1, float(np.std(collection[i][0])),
                                       delta=0.2,
                                       msg=("Expected scale 1 but got {}"
                                            .format(np.std(collection[i][0]))))

    def test_get_scale_statistics(self):
        df = self.spark.createDataFrame(
            [[i, j] for i, j in zip(range(10), [1]*10)], ["c1", "c2"])
        ng = metrics.NeighborhoodGenerator(inputCols=["c1", "c2"])
        actual_scales, actual_means = ng._get_scale_statistics(df)
        self.assertEqual(4.5, actual_means[0], "Wrong mean. Expected 4.5, got {}"
                         .format(actual_means[0]))
        self.assertEqual(1, actual_means[1], "Wrong mean. Expected 1, got {}"
                         .format(actual_means[1]))
        self.assertEqual(1, actual_scales[1],
                         "Wrong scaler factor (stdev). Expected 1, got {}"
                         .format(actual_scales[1]))
        self.assertEqual(3.0277, round(actual_scales[0], 4),
                         "Wrong scaler factor (stdev). Expected 3.0277, got {}"
                         .format(actual_scales[0]))

    def test_get_feature_freqs(self):
        df = self.spark.createDataFrame(
            [[i, j] for i, j in zip([1]*5 + [0]*5, [1]*10)], ["c1", "c2"])
        ng = metrics.NeighborhoodGenerator(inputCols=["c1", "c2"])
        actual_freqs = ng._get_feature_freqs(df)
        self.assertEqual(2, len(actual_freqs.keys()))
        self.assertDictEqual({1: 0.5, 0: 0.5}, actual_freqs["c1"],
                             "Expected 0.5 frequencies for both values. Got"
                             "{}".format(actual_freqs["c1"]))
        self.assertDictEqual({1: 1}, actual_freqs["c2"],
                             "Expected frequency of 1. Got {}"
                             .format(actual_freqs["c2"]))

    def test_neighborhood_generator(self):
        pdf = pd.DataFrame({"f0": np.random.normal(0, 1, 100),
                            "f1": np.random.normal(100, 10, 100),
                            "f2": [0]*80 + [1]*20})
        df = self.spark.createDataFrame(pdf)
        discretizer = QuartileDiscretizer(df, ["f2"], df.columns)
        disc_df = discretizer.discretize(df)

        disc_nGenerator = metrics.NeighborhoodGenerator(
            inputCols=df.columns, neighborhoodSize=100,
            discretizer=discretizer, seed=42)
        undisc_nGeneratorInstance = metrics.NeighborhoodGenerator(
            inputCols=df.columns, neighborhoodSize=100,
            seed=42, sampleAroundInstance=True, categoricalCols=["f2"])
        undisc_nGeneratorMean = metrics.NeighborhoodGenerator(
            inputCols=df.columns, neighborhoodSize=100,
            seed=42, sampleAroundInstance=False, categoricalCols=["f2"])

        with self.subTest("Expected values returned for discretized data."):

            actual = disc_nGenerator.transform(disc_df)
            actual.show(truncate=False)

        with self.subTest("Expected values returned for undiscretized data."):
            pass
