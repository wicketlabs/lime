from pyspark.ml.tests import SparkSessionTestCase
from sparkling_lime.lime_tabular import ReusedEstimator, LocalLinearLearner
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from sparkling_lime.metrics import NeighborhoodGenerator
from pyspark.ml.classification import LogisticRegression


class TestReusedEstimator(SparkSessionTestCase):
    def test_fit_multiple_datasets(self):
        mre = MockReusedLinReg()
        data = [(0, Vectors.dense([-1.0, -1.0]),),
                (0, Vectors.dense([-1.0, 1.0]),),
                (1, Vectors.dense([1.0, -1.0]),),
                (1, Vectors.dense([1.0, 1.0]),)]
        data2 = [(1, Vectors.dense([-1.0, -1.0]),),
                 (1, Vectors.dense([-1.0, 1.0]),),
                 (0, Vectors.dense([1.0, -1.0]),),
                 (0, Vectors.dense([1.0, 1.0]),)]
        dataset = self.spark.createDataFrame(data, ["label", "features"])
        dataset2 = self.spark.createDataFrame(data2, ["label", "features"])
        datasets = [dataset]*5 + [dataset2]*5
        epm = MockReusedLinReg().extractParamMap()
        models = [0]*10
        for index, model in mre.fitMultipleDatasets(mre, datasets, epm):
            models[index] = model
        for index, model in enumerate(models):
            with self.subTest("Model transforms the data"):
                self.assertCountEqual(["label", "features", "prediction"],
                                      model.transform(dataset).columns)
                self.assertEqual("<class 'pyspark.sql.dataframe.DataFrame'>",
                                 str(type(model.transform(dataset))))
        with self.subTest("Model is actually fitted on data "
                          "(not copied instance"):
            correct_transforms = models[0].transform(datasets[0])
            incorrect_transforms = models[0].transform(datasets[-1])
            correct_preds = correct_transforms.filter(
                col("label") == col("prediction")).count()
            incorrect_preds = incorrect_transforms.filter(
                col("label") != col("prediction")).count()
            self.assertEqual(4, correct_preds,
                             ("Expected all predictions to be correct, but "
                              "only {} were.".format(correct_preds)))
            self.assertEqual(4, incorrect_preds,
                             ("Expected all predictions to be incorrect, but "
                              "only {} were.".format(incorrect_preds)))


class TestLocalLinearLearner(SparkSessionTestCase):
    def test_parallel_fitting(self):
        lll = LocalLinearLearner(labelCol="label", featuresCol="features")
        data = [(0, Vectors.dense([-1.0, -1.0]),),
                (0, Vectors.dense([-1.0, 1.0]),),
                (1, Vectors.dense([1.0, -1.0]),),
                (1, Vectors.dense([1.0, 1.0]),)]

        dataset = self.spark.createDataFrame(data, ["label", "features"])
        datasets = [dataset] * 10

        lll.setParallelism(1)
        serial_models = lll.fit(datasets).fittedModels
        lll.setParallelism(2)
        parallel_models = lll.fit(datasets).fittedModels
        self.assertEqual(len(serial_models), len(parallel_models),
                         "Number of models processed serially was not the same"
                         " as the number of models processed in parallel.")


class TestParallelTransformer(SparkSessionTestCase):
    def test_parallel_transform(self):
        lll = LocalLinearLearner(labelCol="label", featuresCol="features")
        data = [(0, Vectors.dense([-1.0, -1.0]),),
                (0, Vectors.dense([-1.0, 1.0]),),
                (1, Vectors.dense([1.0, -1.0]),),
                (1, Vectors.dense([1.0, 1.0]),)]

        dataset = self.spark.createDataFrame(data, ["label", "features"])
        datasets = [dataset] * 10
        model = lll.fit(datasets)
        model.parallelism = 1
        serial_transformed = model.transform(datasets)
        model.parallelism = 2
        parallel_transformed = model.transform(datasets)
        self.assertEqual(len(serial_transformed), len(parallel_transformed),
                         "Number of transformed datasets processed serially"
                         " was not the same as the datasets processed in"
                         " parallel.")
        for s, p in zip(serial_transformed, parallel_transformed):
            self.assertEqual(s.columns, p.columns,
                             "Columns of transformed data processed serially"
                             " was not the same as the datasets processed"
                             " in parallel.")

        parallel_transformed[0].show()
        serial_transformed[0].show()

class MockReusedLinReg(LinearRegression, ReusedEstimator):
    def __init__(self):
        super().__init__()


class TestExplainer(SparkSessionTestCase):

    def _make_data(self, n_rows=50, arr=False):
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
            df = self.spark.createDataFrame(data)
            return df

    def test_explainer(self):
        df = self._make_data()
        df = df.withColumn("label", col("f0"))
        assembler = VectorAssembler(inputCols=["f0", "f1", "f2", "f3"],
                                    outputCol="features")
        df = assembler.transform(df)
        lrm = LogisticRegression().fit(df)
        df = lrm.transform(df)
        ngen = NeighborhoodGenerator(inputCols=["f0", "f1", "f2", "f3"],
                                     inverseOutputCols=["n_f0", "n_f1", "n_f2", "n_f3"],
                                     neighborhoodSize=100,
                                     sampleAroundInstance=False,
                                     categoricalCols=["f0"])





