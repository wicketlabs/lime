from sparkling_lime import metrics
from pyspark.ml.linalg import Vectors
from unittest import TestCase
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct


class TestMetrics(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[4]").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.sparkContext.stop()

    def test_kernel(self):
        pass

    def test_pairwise_distance_vector(self):
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

    def test_pairwise_distance_columns(self):
        data = [(0, Vectors.dense([-1.0, -1.0]), Vectors.dense([-1.0, -1.0]),),
                (1, Vectors.dense([-1.0, 1.0]), Vectors.dense([-1.0, -1.0]),),
                (2, Vectors.dense([1.0, -1.0]), Vectors.dense([-1.0, -1.0]),),
                (3, Vectors.dense([1.0, 1.0]), Vectors.dense([-1.0, -1.0]),)]
        df = self.spark.createDataFrame(data, ["id", "other", "orig"])
        df = df.withColumn("features", struct("orig", "other"))
        d1 = metrics.PairwiseDistance(inputCol="features", outputCol="output",
                                      metric="euclidean")
        actual = d1.transform(df).select("id", "output").collect()
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 2.8284271247461903}
        for r in actual:
            with self.subTest(
                    "Distances are calculated correctly: i={}".format(r["id"])):
                self.assertAlmostEqual(r["output"], expected[r["id"]])

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
