import threading
from pyspark.ml.param import Params



def makeMixinClass(cls, mixin):
    """
    Return a new class that inherits from the given class object and a
    mixin.
    """

    class NewClass(cls, mixin):
        pass

    NewClass.__name__ = "{}_{}".format(cls.__name__, mixin.__name__)
    return NewClass


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