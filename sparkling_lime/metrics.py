"""
Stores methods for calculating metrics
"""
from pyspark.sql.functions import col, sqrt, exp, pow
from pyspark.sql.types import DoubleType
from scipy.spatial import distance
from pyspark.ml import Transformer, UnaryTransformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only


class PairwiseDistance(UnaryTransformer, DefaultParamsReadable,
                       DefaultParamsWritable):
    """
    Calculates pairwise distances between a feature column and a provided
     vector of features, and outputs the distances to a column.
    """

    validMetrics = ["euclidean"]

    rowVector = Param(Params._dummy(), "rowVector",
                      "The denseVector of features by which the values of the"
                      " dataset are compared to calculate pairwise distances.",
                      typeConverter=TypeConverters.toVector)

    metric = Param(Params._dummy(), "metric",
                   ("The distance metric. Supported metrics include: {}"
                    .format(",".join(validMetrics))),
                   typeConverter=TypeConverters.toString)

    _distance_fns = {"euclidean": distance.euclidean}

    @keyword_only
    def __init__(self, rowVector=None, inputCol=None, outputCol=None,
                 metric="euclidean"):
        super(PairwiseDistance, self).__init__()
        self._setDefault(metric="euclidean")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, rowVector=None, inputCol="features",
                  outputCol="distances", metric="euclidean"):
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

    def getMetric(self):
        """
        Gets the value for 'metric' param, or the default.
        """
        return self.getOrDefault(self.metric)

    def setMetric(self, metric):
        """
        Sets the value for 'metric' param.
        """
        self._set(metric=metric)

    def validateInputType(self, inputType):
        """
        Validates the input type. Throw an exception if it is invalid.
        """
        if inputType != VectorUDT():
            raise TypeError("Bad input type: {}. ".format(inputType) +
                            "Requires Double.")

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
        metric = self.getMetric()
        distance_fn = self._distance_fns[metric]
        return lambda x: distance_fn(x, rowVector)


class KernelWeight(HasInputCol, HasOutputCol, Transformer,
                   DefaultParamsReadable, DefaultParamsWritable):
    """
    Transforms a column of distances into a column of proximity values.
    """
    kernelWidth = Param(Params._dummy(), "kernelWidth",
                        "Kernel width to use in the kernel function.",
                        typeConverter=TypeConverters.toFloat)

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

    def getKernelWidth(self):
        """
        Get the value for 'kernelWidth' param, or the default.
        """
        return self.getOrDefault(self.kernelWidth)

    def setRowVector(self, kernelWidth):
        """
        Set the value for 'rowVector' param.
        """
        self._set(kernelWidth=kernelWidth)

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
