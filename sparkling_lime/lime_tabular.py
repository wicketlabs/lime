from sparkling_lime.sparkling_lime_base import BaseSparkMethods
from sparkling_lime.discretize import BaseDiscretizer, QuartileDiscretizer, \
    DecileDiscretizer
from sparkling_lime.metrics import HasKernelWidth, HasDistanceMetric
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sqrt, exp, pow
from pyspark.sql.types import DoubleType
from scipy.spatial import distance
from pyspark.ml import Transformer, UnaryTransformer, Estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasFeaturesCol, \
    HasLabelCol, HasNumFeatures, HasSeed, HasParallelism, HasPredictionCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
from pyspark.ml.regression import LinearRegression
from multiprocessing.pool import ThreadPool
from pyspark.sql import DataFrame
import threading
from pyspark.ml.common import _py2java
from pyspark.ml.wrapper import JavaParams
from pyspark.sql import SparkSession


def _parallelDatasetsFitTasks(wrapper, est, trains, epm):
    """
    Creates a list of callables which can be called from different threads to fit
    estimators in parallel to multiple training sets (`trains`).
    Each callable returns an `index` value and the fitted model.

    :param est: Estimator, the estimator to be fit.
    :param trains: List of DataFrames, (training data set), used for fitting.
    :param epm: ParamMap to be used during fitting
    :return: (int, `Estimator`), an index into `trains` and the associated
    fitted estimator
    """
    modelIter = wrapper.fitMultipleDatasets(est, trains, epm)

    def singleTask():
        index, model = next(modelIter)
        return index, model

    return [singleTask] * len(trains)


def _parallelDatasetsTransformTasks(wrapper, models, datasets):
    """
    Creates a list of callables which can be called from different threads to
    transform multiple datasets by models in parallel.
    Each callable returns an `index` value as well as the transformed data.

    :param wrapper: Instance of the class containing the transformMultipleDatasets
       method.
    :param models: The fitted models to be used to transform teh datasets/
    :param train: DataFrame, training data set, used for fitting.
    :return: (int, `pyspark.sql.DataFrame`), an index into `models`/`datasets`
        and the associated transformed dataset.
    """
    if len(models) != len(datasets):
        raise ValueError("Number of models must equal number of datasets.")

    modelIter = wrapper.transformMultipleDatasets(models, datasets)

    def singleTask():
        index, transformed = next(modelIter)
        return index, transformed

    return [singleTask] * len(models)


class ParallelTransformer(Params):

    def transformMultipleDatasets(self, transformers, datasets):
        """
        Applies transforms each dataset in `datasets` using the associated
        fitted model in `transformers`
        :param transformers: List of fitted models to transform the datasets
        :param datasets: List of datasets to be transformed
        :return: Thread safe iterable which contains one transformed dataset
        for each raw dataset. Each call to `next(modelIterator)` will return
        `(index, dataset)` where data was transfromed using
        `transformers[index]`. `index` values may not be sequential.
        """

        def transformSingleDataset(index):
            model = transformers[index].copy()
            return model.transform(datasets[index])

        return _TransformMultipleDatasetsIterator(transformSingleDataset,
                                                  len(transformers))


class ReusedEstimator(Params):

    def fitMultipleDatasets(self, est, datasets, paramMap):
        """
        Fits a model with a paramMap to each dataset in datasets.

        :param dataset: input datasets, which is an array of
        :py:class:`pyspark.sql.DataFrame`.
        :param paramMaps: A param map used for fitting the model
        :return: Thread safe iterable which contains one model for each dataset.
            Each call to `next(modelIterator)` will return `(index, model)`
            where model was fit using `datasets[index]`.
            `index` values may not be sequential.
        """
        estimator = est.copy()

        def fitSingleModel(index):
            return estimator.fit(datasets[index], paramMap)

        return _FitMultipleDatasetsIterator(fitSingleModel, len(datasets))


class _TransformMultipleDatasetsIterator(object):
    """
    Used by default implementation of Estimator.fitMultiple to produce models in a thread safe
    iterator. This class handles the simple case of fitMultiple where each param map should be
    fit independently.

    :param fitSingleModel: Function: (int => Model) which fits an estimator to a dataset.
        `fitSingleModel` may be called up to `numModels` times, with a unique index each time.
        Each call to `fitSingleModel` with an index should return the Model associated with
        that index.
    :param numModel: Number of models this iterator should produce.

    See Estimator.fitMultiple for more info.
    """
    def __init__(self, transformSingleModel, numModels):
        """

        """
        self.transformSingleModel = transformSingleModel
        self.numModel = numModels
        self.counter = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            index = self.counter
            if index >= self.numModel:
                raise StopIteration("No models remaining.")
            self.counter += 1
        return index, self.transformSingleModel(index)

    def next(self):
        """For python2 compatibility."""
        return self.__next__()


class _FitMultipleDatasetsIterator(object):
    """
    Used by default implementation of Estimator.fitMultiple to produce models in a thread safe
    iterator. This class handles the simple case of fitMultiple where each param map should be
    fit independently.

    :param fitSingleModel: Function: (int => Model) which fits an estimator to a dataset.
        `fitSingleModel` may be called up to `numModels` times, with a unique index each time.
        Each call to `fitSingleModel` with an index should return the Model associated with
        that index.
    :param numModel: Number of models this iterator should produce.

    See Estimator.fitMultiple for more info.
    """
    def __init__(self, fitSingleModel, numModels):
        """

        """
        self.fitSingleModel = fitSingleModel
        self.numModel = numModels
        self.counter = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            index = self.counter
            if index >= self.numModel:
                raise StopIteration("No models remaining.")
            self.counter += 1
        return index, self.fitSingleModel(index)

    def next(self):
        """For python2 compatibility."""
        return self.__next__()


class LocalLinearLearnerParams(Params):
    estimator = Param(Params._dummy(), "estimator",
                      "Regressor to use in explanation. Defaults to ridge"
                      " regression.")
    estimatorParamMap = Param(Params._dummy(), "estimatorParamMap", "")  # TODO

    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`localEstimator`
        """
        return self._set(estimator=value)

    def getEstimator(self):
        """
        Gets the value of localEstimator or its default value.
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


class LocalLinearLearner(HasFeaturesCol, HasLabelCol, HasSeed,
                         Estimator, ReusedEstimator, LocalLinearLearnerParams,
                         DefaultParamsReadable, DefaultParamsWritable,
                         HasParallelism):
    """
    # NOTE: Need pre-weighted and distance-calculated data (is this okay????)
    """

    @keyword_only
    def __init__(self, estimator=None, featuresCol="neighborhood",
                 labelCol="localLabel", parallelism=1, seed=None):
        super(LocalLinearLearner, self).__init__()
        self._setDefault(featuresCol="neighborhood", labelCol="localLabel",
                         parallelism=1,
                         localEstimator=LinearRegression(
                             regParam=1.0, elasticNetParam=0.0,
                             featuresCol=featuresCol, labelCol=labelCol))
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, estimator=None, labelCol="localLabel",
                  featuresCol="neighborhood", parallelism=1):
        """
        Sets params for this KernelWeight transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, datasets):
        est = self.getOrDefault(self.localEstimator)
        epm = est.extractParamMap()
        if isinstance(datasets, DataFrame):
            datasets = [datasets]
        else:
            datasets = list(datasets)
        numModels = len(datasets)
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        models = [0] * len(datasets)
        tasks = _parallelDatasetsFitTasks(self, est, datasets, epm)
        for j, model in pool.imap_unordered(lambda f: f(), tasks):
            models[j] = model
            datasets[j].unpersist()
        llm = self._copyValues(LocalLinearLearnerModel(fittedModels=models))
        return llm


class LocalLinearLearnerModel(HasFeaturesCol, HasLabelCol, HasPredictionCol,
                              HasParallelism, HasSeed, LocalLinearLearnerParams,
                              Transformer, ParallelTransformer):
    fittedModels = Param(Params._dummy(), "fittedModels",
                         "List of models fitted to datasets.")

    def setFittedModels(self, value):
        """
        Sets the value of :py:attr:`fittedModels`
        """
        return self._set(fittedModels=value)

    def getFittedModels(self):
        """
        Gets the value of fittedModels or the default.
        """
        return self.getOrDefault(self.fittedModels)

    @keyword_only
    def __init__(self, globalEstimator=None,
                 featuresCol="features", labelCol="label",
                 localLabelCol="localLabel", parallelism=1,
                 seed=None, fittedModels=()):
        super().__init__()
        self._setDefault(featuresCol="features",
                         labelCol="label", localLabelCol="localLabel",
                         parallelism=1)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, globalEstimator=None,
                  labelCol="label", localLabelCol="localLabel",
                  featuresCol="features",
                  explainLabels=(1,), fittedModels=()):
        """
        Sets params for this KernelWeight transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, datasets):
        models = self.getOrDefault(self.fittedModels)
        if isinstance(datasets, DataFrame):
            datasets = [datasets]
        else:
            datasets = list(datasets)
        numModels = len(models)

        if numModels != len(datasets):
            raise ValueError("Number of models did not match number of datasets"
                             " to transform.")
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        transformed_datasets = [0] * len(datasets)
        tasks = _parallelDatasetsTransformTasks(self, models, datasets)
        for j, transformed_data in pool.imap_unordered(lambda f: f(), tasks):
            transformed_datasets[j] = transformed_data
            datasets[j].unpersist()
        return transformed_datasets
#
#
# class LimeTabularExplainer(BaseSparkMethods):
#     def __init__(self,
#                  training_data,
#                  mode="classification",
#                  training_labels=None,
#                  feature_names=None,
#                  categorical_features=None,
#                  categorical_names=None,
#                  kernel_width=None,
#                  verbose=False,
#                  class_names=None,
#                  feature_selection='auto',
#                  discretize_continuous=True,
#                  discretizer='quartile',
#                  sample_around_instance=False,
#                  random_state=None):
#         """
#
#         :param training_data:
#         :param mode:
#         :param training_labels:
#         :param feature_names:
#         :param categorical_features: Names of categorical feature columns
#         :param categorical_names: -- basically the stringindexer map #TODO
#         :param kernel_width:
#         :param verbose:
#         :param class_names:
#         :param feature_selection:
#         :param discretize_continuous:
#         :param discretizer:
#         :param sample_around_instance:
#         :param random_state:
#         """
#         super().__init__()
#         # TODO
#         # self.random_state = check_random_state(random_state)
#         self.mode = mode
#         self.categorical_features = categorical_features or []
#         self.categorical_names = categorical_names or {}
#         self.sample_around_instance = sample_around_instance
#         self.feature_names = feature_names or []
#         self.discretizer = None
#         if discretize_continuous:
#             if discretizer == 'quartile':
#                 self.discretizer = QuartileDiscretizer(
#                         training_data, self.categorical_features,
#                         self.feature_names, labels=training_labels)
#             elif discretizer == 'decile':
#                 self.discretizer = DecileDiscretizer(
#                         training_data, self.categorical_features,
#                         self.feature_names, labels=training_labels)
#             elif isinstance(discretizer, BaseDiscretizer):
#                 self.discretizer = discretizer
#             else:
#                 raise ValueError(("Discretizer must be 'quartile', 'decile',",
#                                   " or a BaseDiscretizer instance"))
#
#         self.feature_selection = feature_selection
#         self.scaler = None
#         self.class_names = class_names
#         self.scaler.fit(training_data)  # TODO
#         self.feature_values = {}
#         self.feature_frequencies = {}
#
#
# class ExplainerParams(Params):
#     localEstimator = Param(Params._dummy(), "localEstimator",
#                            "Regressor to use in explanation. Defaults to ridge"
#                            " regression.")
#     globalEstimator = Param(Params._dummy(), "globalEstimator",
#                             "Fitted model to generate global predictions on "
#                             "perturbed dataset.")
#     explainLabels = Param(Params._dummy(), "explainLabels",
#                           "Iterable of (numerical) labels to be explained.",
#                           typeConverter=TypeConverters.toListInt)
#     discretizer = Param(Params._dummy(), "discretizer",
#                         ("The discretizer. Supported discretizers are "
#                          "'QuartileDiscretizer', 'DecileDiscretizer' or None."))
#     weightCol = Param(Params._dummy(), "weightCol",
#                       "Name of column with weights.",
#                       typeConverter=TypeConverters.toString)
#     distanceCol = Param(Params._dummy(), "distanceCol",
#                         "Name of column with feature distances.",
#                         typeConverter=TypeConverters.toString)
#     neighborhoodCol = Param(Params._dummy(), "neighborhoodCol",
#                             "Name of column with neighborhood of values "
#                             "sampled around a mean or feature value according "
#                             "to the feature's statistics. Needs to be an array "
#                             "of named structs (inverse, binary).")
#     originalFeaturesCol = Param(Params._dummy(), "originalFeatureCol",
#                                 "")  # TODO
#     localLabelCol = Param(Params._dummy(), "localLabelCol", "")  # TODO
#     estimatorParamMap = Param(Params._dummy(), "estimatorParamMap", "")  # TODO
#
#     def setLocalEstimator(self, value):
#         """
#         Sets the value of :py:attr:`localEstimator`
#         """
#         return self._set(localEstimator=value)
#
#     def getLocalEstimator(self):
#         """
#         Gets the value of localEstimator or its default value.
#         """
#         return self.getOrDefault(self.localEstimator)
#
#     def setNeighborhoodCol(self, value):
#         """
#         Sets the value of :py:attr: `neighborhoodCol`
#         """
#         return self._set(neighborhoodCol=value)
#
#     def getNeighborhoodCol(self):
#         """
#         Gets the value of neighborhoodCol or its default value.
#         """
#         return self.getOrDefault(self.neighborhoodCol)
#
#     def setEstimatorParamMap(self, value):
#         """
#         Sets the value of :py:attr:`estimatorParamMap`.
#         """
#         return self._set(estimatorParamMaps=value)
#
#     def getEstimatorParamMap(self):
#         """
#         Gets the value of estimatorParamMap or its default value.
#         """
#         return self.getOrDefault(self.estimatorParamMap)



# Need to generate the perturbed dataset
# Then add the distances
# Then add the weights
# Then train a linear model
# Then transform and score