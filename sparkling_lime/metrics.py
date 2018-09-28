"""
Stores methods for calculating metrics
"""
from pyspark.sql.functions import col, sqrt, exp, pow, lit
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from scipy.spatial import distance
from pyspark.ml import Transformer, UnaryTransformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCols, HasInputCol, HasOutputCol, HasSeed
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark import keyword_only
import numpy as np
from numpy.random import RandomState


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
        self._setDefault(metric="euclidean", rowVector=None)
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


class NeighborhoodGenerator(HasInputCols, HasOutputCol, HasSeed, Transformer,
                            DefaultParamsWritable, DefaultParamsReadable):
    neighborhoodSize = Param(Params._dummy(), "neighborhoodSize",
                             ("The size of the neighborhood to generate; the  "
                              "number of perturbed samples to generate."),
                             typeConverter=TypeConverters.toInt)
    discretizer = Param(Params._dummy(), "discretizer",
                        ("A fitted QuartileDiscretizer or DecileDiscretizer "
                         "(optional)"))
    format = Param(Params._dummy(), "format",
                   ("The format to use ('wide'/'narrow'. If 'narrow', the "
                    "neighborhood will be output as a single 2darray column "
                    "`outputCol` of size [neighborhoodSize][i], where `i` is "
                    "the number of columns in the `inputCols` list. If "
                    "'wide', the neighborhood will be output as `i` 1darray "
                    "columns of length `neighborhoodSize`, named sequentially "
                    "(i.e. <outputCol>_0, <outputCol>_1, ..., <outputCol>_n)"))
    sampleAroundInstance = Param(Params._dummy(), "sampleAroundInstance",
                                 ("If True, will sample continuous features in"
                                  " in perturbed samples from a normal centered"
                                  " at the instance being explained. Otherwise,"
                                  " the normal is centered on the mean of the"
                                  " feature data."),
                                 typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None, neighborhoodSize=5000,
                 discretizer=None, seed=None, sampleAroundInstance=False):
        super(NeighborhoodGenerator, self).__init__()
        self._setDefault(neighborhoodSize=5000, discretizer=None,
                         sampleAroundInstance=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None, neighborhoodSize=5000,
                  discretizer=None, sampleAroundInstance=False, seed=None):
        """
        Sets params for this PairwiseEuclideanDistance transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setNeighborhoodSize(self, value):
        """
        Sets the value of :py:attr:neighborhoodSize
        """
        return self._set(neighborhoodSize=value)

    def getNeighborhoodSize(self):
        """
        Returns the value of neighborhoodSize or the default value.
        """
        return self._getOrDefault(self.neighborhoodSize)

    def setDiscretizer(self, value):
        """
        Sets the value of :py:attr:discretizer
        """
        return self._set(discretizer=value)

    def getDiscretizer(self):
        """
        Returns the value of discretizer or the default value.
        """
        return self._getOrDefault(self.discretizer)

    def setFormat(self, value):
        """
        Sets the value of :py:attr:format
        """
        return self._set(format=value)

    def getFormat(self):
        """
        Returns the value of format or the default value.
        """
        return self._getOrDefault(self.format)

    def setSampleAroundInstance(self, value):
        """
        Sets the value of :py:attr:sampleAroundInstance
        """
        return self._set(sampleAroundInstance=value)

    def getSampleAroundInstance(self):
        """
        Returns the value of sampleAroundInstance or the default value.
        """
        return self._getOrDefault(self.sampleAroundInstance)

    @staticmethod
    def _make_zeroes(x, y, format):
        """
        Make a column of 2darray of zeros[x][y]
        :param x: number of rows
        :param y: number of columns
        :return: `pyspark.sql.Column` of 2darray of zeros, in 1 column if
            "narrow" format, and in `y` columns in "wide" format (named
            numerically from [0,`y`-1])
        """
        if format == "narrow":
            return F.array(*[F.array(*[lit(0)]*y)]*x)
        elif format == "wide":
            return [F.array([lit(0)] * x)] * y

    @staticmethod
    def _make_normals(x, y, format, scale, mean=(), cols=None, seed=None):
        """
        Make a column of 2darray of zeros[x][y]
        :param x: number of rows
        :param y: number of columns
        :param scale: Iterable of len(`y`), of values by which to scale the
            normal distribution (square root of the feature variance)
        :param mean: Iterable of len(`y`) , of values values to add to the
        scaled normal distribution, to center around (means of each feature)
        :param cols: Values of columns to center the normal distribution around
            instead of the mean. Only relevant if a value for `mean` is not
            provided.
        :return: `pyspark.sql.Column` of centered and scaled normal
            distribution, in 1 column if "narrow" format, and in `y` columns
            in "wide" format (named numerically from [0,`y`-1])
        """
        if mean:
            center = mean
        else:
            center = cols

        # Generate [x]*[y] samples of normal distribution, scale each value by
        #   scaling factor and then add the center value
        # For example, let `n` be a random sample from normal distribution,
        #   `s` be the scaling factor for a feature, and `m` the mean for the
        #    feature:
        #       return (n*s) + m, for all s in scale and m in mean.
        # So if x=2 and y=2:
        #    return [[n1*s1 + m1, n2*s2 + m2], [n1`*s1 + m1, n2`*s2 + m2]]
        if format == "narrow":
            return F.array(
                *[F.array(
                    [randn * scaler + val
                     for randn, scaler, val
                     in zip(
                         [F.randn(seed=seed) for i in range(y)], scale, center)
                     ])
                  for i in range(x)])
        # Similar to above, but instead of nested 2darray, what would be the
        # nesting structure is distributed over columns
        # If use above example, x=2 & y=2,
        #    return two columns: [n1*s1 + m1, n2*s2+m2]
        #                      & [n1`*s1 + m1, n2`*s2 + m1]
        elif format == "wide":
            return [F.array(
                        [randn * scaler + val
                            for randn, scaler, val
                            in zip(
                                [F.randn(seed=seed) for i in range(x)], scale, center)
                         ]
                    )
                    for i in range(y)]

    def _get_statistics(self, dataset):
        """
        Gets the relative scaling factor and means of each feature.
        """
        ic = self.getOrDefault(self.inputCols)
        assembler = VectorAssembler(inputCols=ic, outputCol="features")
        assembled_data = assembler.transform(dataset)
        scaler = StandardScaler(inputCol="features", outputCol="scaled")
        scaler_model = scaler.fit(assembled_data)
        scales = [s if s != 0 else 1 for s in scaler_model.std]
        means = [m for m in scaler_model.mean]
        return scales, means

    # TODO: Decide on a format and get rid of the other one
    def _transform(self, dataset):
        ic = self.getOrDefault(self.inputCols)
        oc = self.getOrDefault(self.outputCol)
        discretizer = self.getOrDefault(self.discretizer)
        num_samples = self.getOrDefault(self.neighborhoodSize)
        num_feats = len(ic)
        format = self.getOrDfeault(self.format)
        seed = self.getOrDefault(self.seed)
        sample_around_instance = self.getOrDefault(self.sampleAroundInstance)
        scales, means = self._get_statistics(dataset)

        # Generate a 2d array of zeros or samples from normal distribution,
        #   size=[numSamples][numFeatures]
        if discretizer:
            if format == "narrow":
                dataset = dataset.withColumn(
                    oc,
                    self._make_zeroes(num_samples, num_feats, format="narrow"))
            elif format == "wide":
                wide_cols = self._make_zeroes(num_samples, num_feats,
                                              format="wide")
                wide_cols_named = [c.alias("{}_{}".format(oc, n))
                                   for c, n in zip(wide_cols, range(num_feats))]
                dataset = dataset.select("*", *wide_cols_named)
        else:
            if sample_around_instance:     # Center around feature mean or value
                cols = [col(c) for c in ic]
                means = None
            else:
                cols = None

            if format == "narrow":
                dataset = dataset.withColumn(
                    oc,
                    self._make_normals(num_samples, num_feats, scale=scales,
                                       mean=means, cols=cols, format="narrow",
                                       seed=seed))
            elif format == "wide":
                wide_cols = self._make_normals(num_samples, num_feats,
                                               scale=scales, mean=means,
                                               cols=cols, format="wide",
                                               seed=seed)
                wide_cols_named = [c.alias("{}_{}".format(oc, n))
                                   for c, n in zip(wide_cols, range(num_feats))]
                dataset = dataset.select("*", *wide_cols_named)
        return dataset
