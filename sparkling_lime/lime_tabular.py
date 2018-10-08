from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasFeaturesCol, \
    HasLabelCol, HasSeed, HasParallelism, HasPredictionCol
from pyspark.ml.param import Param, Params
from pyspark import keyword_only
from pyspark.ml.regression import LinearRegression
from multiprocessing.pool import ThreadPool
from pyspark.sql import DataFrame
import threading


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
    Used by default implementation of Estimator.fitMultiple to produce models in
    a thread safe iterator. This class handles the simple case of
    transformMultipleDatasets where each dataset and model should be
    transformed independently.

    :param transformSingleModel: Function: (int => Model) which calls the
    `transform` method on a dataset. `transformSingleDataset` may be called up
    to `numModels` times, with a unique index each time.  Each call to
    `transformSingleDataset` with an index should return the transformed dataset
    associated with that index, created by the fitted model associated with
    that index.
    :param numModel: Number of models this iterator should use. Should be the
    same as the number of transformed datasets desired.

    See ParallelTransformer.transformMultipleDatasets for more info.
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
    Used by default implementation of Estimator.fitMultipleDatasests to produce
    models in a thread safe iterator. This class handles the simple case of
    fitMultipleDatasets where each dataset should be used to fit each model
    independently.

    :param fitSingleModel: Function: (int => Model) which fits an estimator
    to a dataset.
        `fitSingleModel` may be called up to `numModels` times, with a unique
        index each time.
        Each call to `fitSingleModel` with an index should return the Model
        associated with that index.
    :param numModel: Number of models this iterator should produce.

    See ReusedEstimator.fitMultipleDatasets for more info.
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


class LocalLinearLearner(HasFeaturesCol, HasLabelCol, HasSeed,
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
    @keyword_only
    def __init__(self, fittedModels=(), parallelism=1):
        super().__init__()
        self.parallelism = parallelism
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
            transformed_datasets[j] = transformed_data
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
