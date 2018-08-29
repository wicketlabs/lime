"""
Stores methods for calculating metrics
"""
from pyspark.sql.functions import col, udf, sqrt, exp, pow
from pyspark.sql.types import DoubleType
from scipy.spatial import distance
from pyspark.ml import Transformer, UnaryTransformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only


class PairwiseEuclideanDistance(UnaryTransformer, DefaultParamsReadable,
                                DefaultParamsWritable):

    rowVector = Param(Params._dummy(), "rowVector",
                      "The denseVector of features by which the values of the"
                      " dataset are compared to calculate pairwise distances.",
                      typeConverter=TypeConverters.toVector)

    @keyword_only
    def __init__(self, rowVector=None, inputCol=None, outputCol=None):
        super(PairwiseEuclideanDistance, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, rowVector=None, inputCol="features",
                  outputCol="distances"):
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

    def validateInputType(self, inputType):
        if inputType != VectorUDT():
            raise TypeError("Bad input type: {}. ".format(inputType) +
                            "Requires Double.")

    def outputDataType(self):
        return DoubleType()

    def createTransformFunc(self):
        rowVector = self.getRowVector()
        return lambda x: distance.euclidean(x, rowVector)


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

