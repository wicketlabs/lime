"""
Stores methods for calculating metrics
"""
from pyspark.sql.functions import col, sqrt, exp, pow
from pyspark.sql.types import DoubleType
from scipy.spatial import distance
from pyspark.ml import Transformer, UnaryTransformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCols, HasInputCol, HasOutputCol, HasSeed
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only


class HasDistanceMetric(Params):
    """
    Mixin for param distanceMetric: pairwise distance metric
    """
    validDistanceMetrics = ["euclidean"]
    distanceMetric = Param(Params._dummy(), "distanceMetric",
                   ("The distance metric. Supported distanceMetrics include: {}"
                    .format(",".join(validDistanceMetrics))),
                   typeConverter=TypeConverters.toString)
    _distance_fns = {"euclidean": distance.euclidean}

    def __init__(self):
        super(HasDistanceMetric, self).__init__()
        self._setDefault(distanceMetric="euclidean")

    def setDistanceMetric(self, value):
        """
        Sets the value of :py:attr:`distanceMetric`.
        """
        self._set(distanceMetric=value)

    def getDistanceMetric(self):
        """
        Gets the value of distanceMetric or its default value.
        """
        return self.getOrDefault(self.distanceMetric)


class HasKernelWidth(Params):
    """
    Mixin for param kernelWidth: kernel width
    """
    kernelWidth = Param(Params._dummy(), "kernelWidth",
                        "Kernel width to use in the kernel function.",
                        typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasKernelWidth, self).__init__()

    def setKernelWidth(self, value):
        """
        Sets the value of :py:attr:`kernelWidth`
        """
        return self._set(kernelWidth=value)

    def getKernelWidth(self):
        """
        Get the value of kernelWidth or its default value.
        """
        return self.getOrDefault(self.kernelWidth)


class PairwiseDistance(HasDistanceMetric, UnaryTransformer,
                       DefaultParamsReadable, DefaultParamsWritable):
    """
    Calculates pairwise distances between a feature column and a provided
     vector of features, and outputs the distances to a column.
    """
    rowVector = Param(Params._dummy(), "rowVector",
                      "The denseVector of features by which the values of the"
                      " dataset are compared to calculate pairwise distances."
                      " If not provided, will check if inputCol is a struct of"
                      " two vectors, (first, second), and if so will calculate"
                      " the distance from second to first.",
                      typeConverter=TypeConverters.toVector)

    @keyword_only
    def __init__(self, rowVector=None, inputCol=None, outputCol=None,
                 distanceMetric="euclidean"):
        super(PairwiseDistance, self).__init__()
        self._setDefault(distanceMetric="euclidean", rowVector=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, rowVector=None, inputCol="features",
                  outputCol="distances", distanceMetric="euclidean"):
        """
        Sets params for this PairwiseEuclideanDistance transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getRowVector(self):
        """
        Gets the value for 'rowVector' param, or the default.
        """
        return self.getOrDefault(self.rowVector)

    def setRowVector(self, rowVector):
        """
        Sets the value for 'rowVector' param.
        """
        self._set(rowVector=rowVector)

    def validateInputType(self, inputType):
        """
        Validates the input type. Throw an exception if it is invalid.
        """
        if not self.getOrDefault(self.rowVector):
            try:
                numFields = len(inputType.fields)
                if numFields != 2:
                    raise TypeError("Bad input fields. If no rowVector is given"
                                    ", must have two fields. Got {}."
                                    .format(numFields))
                for field in range(numFields):
                    if inputType[field].dataType != VectorUDT():
                        if inputType[field] != VectorUDT():
                            raise TypeError(
                                "Bad input type: {}. Requires Vector."
                                .format(inputType[field]))
            except AttributeError:
                raise TypeError("Bad input type: {}. "
                                "Requires struct of vectors."
                                .format(inputType))
        else:
            if inputType != VectorUDT():
                raise TypeError("Bad input type: {}. ".format(inputType) +
                                "Requires Vector.")

    def outputDataType(self):
        """
        Returns the data type of the output column.
        """
        return DoubleType()

    def createTransformFunc(self):
        """
        Creates the transform function using given params.
        The transform function calculates the pairwise distances between
        rows of features based on the given distance metric.
        """
        rowVector = self.getRowVector()
        metric = self.getDistanceMetric()
        distance_fn = self._distance_fns[metric]
        if rowVector:
            return lambda x: distance_fn(x, rowVector)
        else:
            return lambda x: distance_fn(x[1], x[0])


class KernelWeight(HasKernelWidth, HasInputCol, HasOutputCol, Transformer,
                   DefaultParamsReadable, DefaultParamsWritable):
    """
    Transforms a column of distances into a column of proximity values.
    """

    @keyword_only
    def __init__(self, kernelWidth=None, inputCol=None, outputCol=None):
        super(KernelWeight, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, kernelWidth=None, inputCol="features",
                  outputCol="distances"):
        """
        Sets params for this KernelWeight transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        """
        Transforms the input dataset.

        :param dataset: input dataset, which is an instance of
         :py:class:`pyspark.sql.DataFrame`
        :returns: transformed dataset
        """
        kernelWidth = self.getKernelWidth()
        outputCol = self.getOutputCol()
        inputCol = self.getInputCol()
        return dataset.withColumn(
            outputCol,
            sqrt(exp(-(pow(col(inputCol), 2)) / kernelWidth ** 2)))


