from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasFeaturesCol, \
    HasLabelCol, HasSeed, HasParallelism, HasPredictionCol, HasWeightCol
from pyspark.ml.param import Param, Params
from pyspark import keyword_only
from pyspark.ml.regression import LinearRegression
from multiprocessing.pool import ThreadPool
from pyspark.sql import DataFrame
import threading
from pyspark.sql.functions import lit, col
import pyspark.sql.functions as F
from sparkling_lime.sparkling_lime_base import BaseSparkMethods
from sparkling_lime.discretize import DecileDiscretizer, QuartileDiscretizer
from sparkling_lime.metrics import NeighborhoodGenerator, KernelWeight, PairwiseDistance
from pyspark.ml.feature import VectorAssembler
from math import sqrt
from sparkling_lime.utils import _parallelDatasetsTransformTasks, \
    _parallelDatasetsFitTasks, ReusedEstimator, ParallelTransformer


class LocalLinearLearnerParams(Params):
    estimator = Param(Params._dummy(), "estimator",
                      "Regressor to use in explanation. Defaults to ridge"
                      " regression.")
    estimatorParamMap = Param(Params._dummy(), "estimatorParamMap", "")  # TODO

    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`
        """
        return self._set(estimator=value)

    def getEstimator(self):
        """
        Gets the value of estimator or its default value.
        """
        return self.getOrDefault(self.estimator)

    def setEstimatorParamMap(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMap`.
        """
        return self._set(estimatorParamMaps=value)

    def getEstimatorParamMap(self):
        """
        Gets the value of estimatorParamMap or its default value.
        """
        return self.getOrDefault(self.estimatorParamMap)


class LocalLinearLearner(HasFeaturesCol, HasLabelCol, HasSeed, HasWeightCol,
                         Estimator, ReusedEstimator, LocalLinearLearnerParams,
                         HasParallelism):
    """
    """

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMap=None,
                 featuresCol="neighborhood", labelCol="localLabel",
                 parallelism=1, seed=None):
        super().__init__()
        self._setDefault(featuresCol="neighborhood", labelCol="localLabel",
                         parallelism=1, estimatorParamMap=None,
                         estimator=LinearRegression(
                             regParam=1.0, elasticNetParam=0.0,
                             featuresCol=featuresCol, labelCol=labelCol))
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, estimator=None, estimatorParamMap=None,
                  labelCol="localLabel", featuresCol="neighborhood",
                  parallelism=1, seed=None):
        """
        Sets params for this KernelWeight transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, datasets):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMap)
        parallelism = self.getOrDefault(self.parallelism)
        if isinstance(datasets, DataFrame):
            datasets = [datasets]
        else:
            datasets = list(datasets)
        numModels = len(datasets)
        pool = ThreadPool(processes=min(parallelism, numModels))
        models = [0] * len(datasets)
        tasks = _parallelDatasetsFitTasks(self, est, datasets, epm)
        for j, model in pool.imap_unordered(lambda f: f(), tasks):
            models[j] = model
            datasets[j].unpersist()
        llm = self._copyValues(LocalLinearLearnerModel(fittedModels=models,
                                                       parallelism=parallelism))
        return llm

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying estimator,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        newLLL = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newLLL.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMap remains the same
        return newLLL


class LocalLinearLearnerModel(HasFeaturesCol, HasLabelCol, HasPredictionCol,
                              HasSeed,
                              Transformer, ParallelTransformer):
    """
    NOTE: Currently r2 score ignores weight column. Will be updated with
    future versions of pyspark.
    """
    @keyword_only
    def __init__(self, fittedModels=(), parallelism=1, scoreCol="r2",
                 coeffCol="coeff", interceptCol="intercept"):
        super().__init__()
        self.parallelism = parallelism
        self.scoreCol = scoreCol
        self.coeffCol = coeffCol
        self.interceptCol = interceptCol
        self.fittedModels = list(fittedModels)

    def _transform(self, datasets):
        models = self.fittedModels
        if isinstance(datasets, DataFrame):
            datasets = [datasets]
        else:
            datasets = list(datasets)
        numModels = len(models)

        if numModels != len(datasets):
            raise ValueError("Number of models did not match number of datasets"
                             " to transform.")
        pool = ThreadPool(processes=min(self.parallelism, numModels))
        transformed_datasets = [0] * len(datasets)
        tasks = _parallelDatasetsTransformTasks(self, models, datasets)
        for j, transformed_data in pool.imap_unordered(lambda f: f(), tasks):
            model = self.fittedModels[j]
            model_summary = model.summary
            transformed_datasets[j] = transformed_data\
                .withColumn(self.scoreCol, lit(model_summary.r2))\
                .withColumn(self.coeffCol,
                            F.array(*[lit(c) for c in model.coefficients]))\
                .withColumn(self.interceptCol, lit(model.intercept))
            datasets[j].unpersist()
        return transformed_datasets

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying estimator,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        fittedModels = [m.copy(extra) for m in self.fittedModels]
        parallelism = self.parallelism
        return LocalLinearLearnerModel(fittedModels, parallelism)
