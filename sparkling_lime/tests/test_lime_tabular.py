from pyspark.ml.tests import SparkSessionTestCase
from pyspark.ml.base import Estimator
from sparkling_lime.lime_tabular import ReusedEstimator, LocalLinearLearner
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col


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
        lll = LocalLinearLearner()
        data = [(0, Vectors.dense([-1.0, -1.0]),),
                (0, Vectors.dense([-1.0, 1.0]),),
                (1, Vectors.dense([1.0, -1.0]),),
                (1, Vectors.dense([1.0, 1.0]),)]

        dataset = self.spark.createDataFrame(data, ["label", "features"])
        datasets = [dataset] * 10

        lll.setParallelism(1)
        serial_models = lll.fit(datasets).getFittedModels()
        lll.setParallelism(2)
        parallel_models = lll.fit(datasets).getFittedModels()
        self.assertEqual(len(serial_models), len(parallel_models),
                         "Number of models processed serially was not the same"
                         " as the number of models processed in parallel.")


class TestParallelTransformer(SparkSessionTestCase):
    def test_parallel_transform(self):
        lll = LocalLinearLearner()
        data = [(0, Vectors.dense([-1.0, -1.0]),),
                (0, Vectors.dense([-1.0, 1.0]),),
                (1, Vectors.dense([1.0, -1.0]),),
                (1, Vectors.dense([1.0, 1.0]),)]

        dataset = self.spark.createDataFrame(data, ["label", "features"])
        datasets = [dataset] * 10
        model = lll.fit(datasets)
        model.setParallelism(1)
        serial_transformed = model.transform(datasets)
        model.setParallelism(2)
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

class MockReusedLinReg(LinearRegression, ReusedEstimator):
    def __init__(self):
        super().__init__()

