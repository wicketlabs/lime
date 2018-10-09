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
from pyspark.ml.feature import VectorAssembler, StandardScaler, Bucketizer
from pyspark import keyword_only
from pyspark.ml import Pipeline


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


class NeighborhoodGeneratorParams(Params):
    """
    Private mixin class for tracking params for NeighborhoodGenerator
    """
    neighborhoodSize = Param(Params._dummy(), "neighborhoodSize",
                             ("The size of the neighborhood to generate; the  "
                              "number of perturbed samples to generate."),
                             typeConverter=TypeConverters.toInt)
    discretizer = Param(Params._dummy(), "discretizer",
                        ("A fitted QuartileDiscretizer or DecileDiscretizer "
                         "(optional)"))
    format = Param(Params._dummy(), "format",
                   ("The format to use ('wide'/'narrow'). If 'narrow', the "
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
    categoricalCols = Param(Params._dummy(), "categoricalCols",
                            ("Names of categorical columns (only relevant if "
                             "'discretizer' is None."),
                            typeConverter=TypeConverters.toListString)
    binaryOutputCols = Param(Params._dummy(), "binaryOutputCols",
                             ("Names of columns corresponding to the 'binary' "
                              "neighborhood generated around that feature. "
                              "By default, will prefix the names of the "
                              "inputCols with 'binary_'."),
                             typeConverter=TypeConverters.toListString)
    inverseOutputCols = Param(Params._dummy(), "inverseOutputCols",
                              ("Names of the columns corresponding to the "
                               "neighborhood generated around that feature by "
                               "perturbing feature values (inverse of mean-"
                               "centering and scaling. By default, will prefix "
                               "the names of the inputCols with 'inverse_'."),
                              typeConverter=TypeConverters.toListString)

    def __init__(self):
        super(NeighborhoodGeneratorParams, self).__init__()

    def setNeighborhoodSize(self, value):
        """
        Sets the value of :py:attr:neighborhoodSize
        """
        return self._set(neighborhoodSize=value)

    def getNeighborhoodSize(self):
        """
        Returns the value of neighborhoodSize or the default value.
        """
        return self.getOrDefault(self.neighborhoodSize)

    def setDiscretizer(self, value):
        """
        Sets the value of :py:attr:discretizer
        """
        return self._set(discretizer=value)

    def getDiscretizer(self):
        """
        Returns the value of discretizer or the default value.
        """
        return self.getOrDefault(self.discretizer)

    def setFormat(self, value):
        """
        Sets the value of :py:attr:format
        """
        return self._set(format=value)

    def getFormat(self):
        """
        Returns the value of format or the default value.
        """
        return self.getOrDefault(self.format)

    def setSampleAroundInstance(self, value):
        """
        Sets the value of :py:attr:sampleAroundInstance
        """
        return self._set(sampleAroundInstance=value)

    def getSampleAroundInstance(self):
        """
        Returns the value of sampleAroundInstance or the default value.
        """
        return self.getOrDefault(self.sampleAroundInstance)

    def setCategoricalCols(self, value):
        """
        Sets the value of :py:attr:categoricalCols
        """
        return self._set(categoricalCols=value)

    def getCategoricalCols(self):
        """
        Returns the value of categoricalCols or the default value.
        """
        return self.getOrDefault(self.categoricalCols)

    def setInverseOutputCols(self, value):
        """
        Sets the value of :py:attr:inverseOutputCols
        """
        return self._set(inverseOutputCols=value)

    def getInverseOutputCols(self):
        """Returns the value of inverseOutputCols or the default value."""
        return self.getOrDefault(self.inverseOutputCols)


class NeighborhoodGenerator(Transformer, HasInputCols,
                            HasOutputCol, HasSeed, NeighborhoodGeneratorParams):
    """
    Generate a neighborhood around features.
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None, neighborhoodSize=5000,
                 discretizer=None, seed=None, sampleAroundInstance=False,
                 categoricalCols=None):
        super(NeighborhoodGenerator, self).__init__()
        self._setDefault(inputCols=None, neighborhoodSize=5000, discretizer=None,
                         sampleAroundInstance=False, categoricalCols=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None, neighborhoodSize=5000,
                  discretizer=None, sampleAroundInstance=False, seed=None,
                  categoricalCols=None):
        """
        Sets params for this PairwiseEuclideanDistance transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @staticmethod
    def _make_bins(freqs):
        """
        Create splits from feature frequencies to use in
        `pyspark.ml.feature.Bucketizer` instances.
        :param freqs: List of (value, frequency) tuples
        :return: Tuple of list of bins for a bucketizer and bin: value dictionary
        """
        bins = [0]
        bin_map = {}
        sorted_freqs = sorted(freqs, key=lambda x: x[1])
        for i, f in enumerate(sorted_freqs):
            bins.append(bins[i] + f[1])
            bin_map.update({i: f[0]})
        return bins, bin_map

    @staticmethod
    def zip_explode(*colnames, n):
        """
        Zips array columns together intro a struct col with elements that can
        then be selected individually. Array columns must have a fixed, known
        size and be equal to each other.
        Given a dataframe df:
            +---+------------+------------+
            | id|          c1|          c2|
            +---+------------+------------+
            |  1|[a, b, c, d]|[1, 2, 3, 4]|
            +---+------------+------------+
        >>> df.withColumn("tmp",
        >>>               NeighborhoodGenerator.zip_explode("c1", "c2", n=4))\
        >>>     .select("id", "tmp.c1", "tmp.c2").drop("tmp")\
        >>>     .show()
            +---+---+---+
            | id| c1| c2|
            +---+---+---+
            |  1|  a|  1|
            |  1|  b|  2|
            |  1|  c|  3|
            |  1|  d|  4|
            +---+---+---+
        """
        return F.explode(F.array(*[
            F.struct(*[col(c).getItem(i).alias(c) for c in colnames])
            for i in range(n)
        ]))

    @staticmethod
    def _make_random_choices(dataset, input_col, num_samples, feature_rate_dict,
                             output_col="neighborhood", seed=None):
        """
        Make an array of size `num_samples`, that are the result of
        random sampling from all possible categorical feature values at the
        rates they are present in the feature data.
        :param dataset: `pyspark.sql.DataFrame` of features in wide format
        :param input_col: Name of the input column to generate samples from
        :param num_samples: Number of samples to generate
        :param feature_rate_dict: Dictionary of feature values to rates the
        value occured in the dataset
        :param seed: Random seed
        :return: dataset with output_col with an array of num_samples
        """
        ordered_rates = [(k, v) for k, v in feature_rate_dict.items()]    # For safe iteration
        bins, bin_map = NeighborhoodGenerator._make_bins(ordered_rates)
        orig_cols = dataset.columns
        cols = [F.rand().alias("inverse_{}_{}".format(input_col, i))
                for i in range(num_samples)]
        dataset = dataset.select(cols + orig_cols)
        colnames = ["inverse_{}_{}".format(input_col, i)
                    for i in range(num_samples)]
        bucketed_colnames = ["bucketed_"+c for c in colnames]
        bucketizers = [Bucketizer(inputCol=ic, outputCol="bucketed_"+ic,
                                  splits=bins)
                       for ic in colnames]
        pipeline = Pipeline(stages=bucketizers)
        dataset = pipeline.fit(dataset).transform(dataset)\
            .replace(bin_map, subset=bucketed_colnames)\
            .drop(*colnames)
        dataset = dataset.withColumn(output_col,
                                     F.array(bucketed_colnames))
        dataset = dataset.select(*orig_cols, output_col)
        return dataset

    @staticmethod
    def _make_normals(x, y, scale, mean=(), cols=None, seed=None, colnames=None):
        """
        Make `y` columns of an array of `x` samples of a normal distribution,
        scaled with `scale` and centered around a mean or the values in
        `cols`.
        For example, let `n` be a random sample from normal distribution,
        `s` be the scaling factor for a feature, and `m` the mean for the
        feature:
              return (n*s) + m
        If x=3 & y=2, return two columns:
            [n1*s1 + m1, n2*s1+m1, n3*s1*m1]
            & [n1`*s2 + m2, n2`*s2 + m2, n3`*s2+m2]
        :param x: number of rows
        :param y: number of columns
        :param scale: Iterable of len(`y`), of values by which to scale the
            normal distribution (square root of the feature variance)
        :param mean: Iterable of len(`y`) , of values values to add to the
        scaled normal distribution, to center around (means of each feature)
        :param cols: Values of columns to center the normal distribution around
            instead of the mean. Only relevant if a value for `mean` is not
            provided.
        :param colnames: Names of the columns to return. If none provided,
        will label columns numerically.
        :return: `pyspark.sql.Column` of centered and scaled normal
            distribution, in `y` columns in "wide" format
            (named numerically from [0,`y`-1])
        """
        if mean:
            center = mean
        else:
            center = cols
        if colnames:
            if len(colnames) != y:
                raise ValueError("Column names must be equal to the number of "
                                 "columns created. Got {} names, needed {}"
                                 .format(len(colnames), y))
        if not colnames:
            colnames = [str(j) for j in range(y)]
        return [F.array(
                    *[F.rand() for i in range(x)]).alias(j)
                for j in colnames]

    def _get_scale_statistics(self, dataset, subset=()):
        """
        Gets the relative scaling factor and means of each feature.
        """
        if subset:
            ic = subset
        else:
            ic = self.getOrDefault(self.inputCols)
        assembler = VectorAssembler(inputCols=ic, outputCol="features")
        assembled_data = assembler.transform(dataset)
        scaler = StandardScaler(inputCol="features", outputCol="scaled")
        scaler_model = scaler.fit(assembled_data)
        scales = [s if s != 0 else 1 for s in scaler_model.std]
        means = [m for m in scaler_model.mean]
        return scales, means

    def _get_feature_freqs(self, dataset, subset=None):
        """
        Returns a dictionary of feature value: frequency for each feature.
        E.g. {feature_name: {val1: count1, val2: count2, ..., val_n: count_n}
        """
        if not subset:
            subset = self.getOrDefault(self.inputCols)
        feature_freqs = {}
        for c in subset:
            value_counts = dict([(row[c], row["count"]) for row in
                                dataset.select(c).cube(c).count()
                                .filter(col(c).isNotNull())
                                .collect()])
            total_count = sum(value_counts.values())
            value_rates = dict((k, v/total_count)
                               for k, v in value_counts.items())
            feature_freqs.update({c: value_rates})
        return feature_freqs

    def _undiscretize_helper(self, dataset, subset, orig_col, discretizer):
        """
        Format the dataset so that it can be undiscretized, then undiscretize
        the variable.
        :param dataset: pyspark.sql.DataFrame
        :param subset: Name of the column to undiscretize
        :param orig_col: Name of the original column that the subset column was
            created from (original feature name)
        :param discretizer: A fitted Discretizer class (Quantile/Decile)
        :returns: pyspark.sql.DataFrame with the sampled `subset` column
            undiscretized. Also removes the "binary" element of the struct and
            replaces it with the undiscretized value (since it's no longer
            relevant as a continuous variable).
        """
        other_cols = [c for c in dataset.columns if c != subset]
        id_col = str(hash("_undiscretize_helper"))
        df = dataset.withColumn(id_col, F.monotonically_increasing_id())
        df.cache()    # Force compute of id
        df = df.withColumn(subset, F.explode(subset))
        means, stds, mins, maxs = ({subset: discretizer.means[orig_col]},
                                   {subset: discretizer.stds[orig_col]},
                                   {subset: discretizer.mins[orig_col]},
                                   {subset: discretizer.maxs[orig_col]})
        df = discretizer.static_undiscretize(df, [subset],
                                             means,
                                             stds,
                                             mins,
                                             maxs,
                                             discretizer.random_state)
        df = df.groupBy(id_col, *other_cols)\
            .agg(F.collect_list(subset)    # Only kept to avoid KeyError
                 .alias(subset))\
            .drop(id_col)
        return df

    def _transform(self, dataset):
        """
        Generates a neighborhood around a prediction.
        For numerical features, perturb them by sampling from a normal
        distribution and doing the inverse operation of mean-centering and
        scaling, according to the means and stds in the training data. For
        categorical feature, perturb by sampling according to the training
        distribution, and making a binary feature that is 1 when the value
        is the same as the instance being explained.

        The neighborhood is an array of structs with the following schema:

        Output schema:
            original dataset + for each feature column:

        root
         |-- <outputCol>: array (nullable = true)
         |    |-- element: struct (containsNull = false)
         |    |    |-- inverse: double (nullable = false)
         |    |    |-- binary: double (nullable = false)

        For continuous variables, the 'binary' element is the same as the
        'inverse' element (since it is only relevant for categoical variables).

        :param dataset: `pyspark.sql.DataFrame` of features in wide format
        (one feature per column)
        :return: Dataset with original + perturbed features in wide format
        """
        # Configure column names and save a map of inputs => outputs
        ic = self.getOrDefault(self.inputCols)
        if self.isSet(self.inverseOutputCols):
            inverse_output_cols = self.getOrDefault(self.inverseOutputCols)
        else:
            inverse_output_cols = ["inverse_" + c for c in ic]
        inverse_col_map = dict(zip(ic, inverse_output_cols))

        # Get necessary vars and pull required stats (freqs, scales, means)
        discretizer = self.getOrDefault(self.discretizer)
        num_samples = self.getOrDefault(self.neighborhoodSize)
        seed = self.getOrDefault(self.seed)
        sample_around_instance = self.getOrDefault(self.sampleAroundInstance)
        if discretizer:
            categorical_cols = ic
            continuous_cols = None
        else:
            categorical_cols = self.getOrDefault(self.categoricalCols)
            continuous_cols = [c for c in ic if c not in categorical_cols]
        feature_freqs = self._get_feature_freqs(dataset,
                                                subset=categorical_cols)
        binary_cols = []
        for c in categorical_cols:
            # Randomly sample according to categorical value frequency
            dataset = self._make_random_choices(
                    dataset, c,  num_samples, feature_freqs[c],
                    inverse_col_map[c])
            dataset.cache()
            # Undiscretize if the column is a discretized continuous feature
            if discretizer:
                if c in discretizer.to_discretize:
                    dataset = self._undiscretize_helper(
                        dataset, inverse_col_map[c], c, discretizer)
                else:
                    binary_cols.append(c)
            dataset.cache()
        if continuous_cols:
            num_feats = len(continuous_cols)
            inverse_subset = [inverse_col_map[c] for c in continuous_cols]
            scales, means = self._get_scale_statistics(dataset,
                                                       subset=continuous_cols)
            if sample_around_instance:     # Center around feature mean or value
                cols = [col(c) for c in continuous_cols]
                means = None
            else:
                cols = None
            wide_cols = self._make_normals(num_samples, num_feats,
                                           scale=scales, mean=means,
                                           cols=cols, seed=seed,
                                           colnames=inverse_subset)
            dataset = dataset.select("*", wide_cols)

        generated_cols = list(inverse_col_map.values())
        dataset = dataset.withColumn(
            "tmp",
            NeighborhoodGenerator.zip_explode(*generated_cols, n=num_samples))\
            .drop(*generated_cols)\
            .select("*", *["tmp.{}".format(c) for c in generated_cols])\
            .drop("tmp")
        # Generate a binary col if necessary
        for c in binary_cols:
            dataset = dataset.withColumn(
                "binary_"+c,
                F.when(col(c) == col(inverse_col_map[c]), 1.0).otherwise(0.0))

        return dataset
