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
        setParams(self, rowVector=None, inputCol=None, outputCol=None)
        Sets params for this PairwiseEuclideanDistance.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getRowVector(self):
        return self.getOrDefault(self.rowVector)

    def setRowVector(self, rowVector):
        self._set(rowVector=rowVector)

    def getMetric(self):
        return self.getOrDefault(self.metric)

    def setMetric(self, metric):
        self._set(metric=metric)

    def validateInputType(self, inputType):
        if inputType != VectorUDT():
            raise TypeError("Bad input type: {}. ".format(inputType) +
                            "Requires Double.")

    def outputDataType(self):
        return DoubleType()

    def createTransformFunc(self):
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
        setParams(self, rowVector=None, inputCol=None, outputCol=None)
        Sets params for this KernelWeighter.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getKernelWidth(self):
        return self.getOrDefault(self.kernelWidth)

    def setRowVector(self, kernelWidth):
        self._set(kernelWidth=kernelWidth)

    def _transform(self, dataset):
        kernelWidth = self.getKernelWidth()
        outputCol = self.getOutputCol()
        inputCol = self.getInputCol()
        return dataset.withColumn(
            outputCol,
            sqrt(exp(-(pow(col(inputCol), 2)) / kernelWidth ** 2)))
