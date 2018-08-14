"""
Discretizers classes, to be used in lime_tabular
"""
import numpy as np
from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, stddev
from pyspark.ml.feature import Bucketizer


# TODO: Move this somewhere else
class BaseSparkMethods(object):
    """
    Stores information about the underlying SparkContext and SparkSession.
    Mixin for Sparkling Lime methods.
    """

    def __init__(self):
        self._sparkSession = None

    @property
    def spark(self):
        """
        Shortcut to use more familiar `spark` syntax instead of `sparkSession`.
        """
        return self.sparkSession

    @property
    def sparkSession(self):
        """
        Returns the user-specified Spark Session or the default.
        """
        if self._sparkSession is None:
            self._sparkSession = SparkSession.builder.getOrCreate()
        return self._sparkSession

    @property
    def sc(self):
        """
        Returns the underlying `SparkContext`.
        """
        return self.sparkSession.sparkContext


class BaseDiscretizer(BaseSparkMethods):
    """
    Abstract class - Build a class that inherits from this class to implement
    a custom discretizer.
    Method bins() is to be redefined in the child class, as it is the actual
    custom part of the discretizer.
    """

    __metaclass__ = ABCMeta  # abstract class

    def __init__(self, data, categorical_feature_names, feature_names,
                 labels=None, random_state=None):
        """Initializer
        Args:
            data: numpy 2d array
            categorical_features: list of names (str) corresponding to the
                categorical columns. These features will not be discretized.
                Everything else will be considered continuous, and will be
                discretized.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
        """
        # Need this to be a list of feature names instead
        super().__init__()
        self.to_discretize = ([x for x in feature_names
                              if x not in categorical_feature_names])
        self.names = {}
        self.bucketizers = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.random_state = check_random_state(random_state)

        bins = self.bins(data, labels)
        bins = [np.unique(x).tolist() for x in bins]

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = len(qts) - 1    # Number of borders (=#bins-1)
            boundaries = (self._calculate_column_statistic(data, feature, "min")[0],
                          self._calculate_column_statistic(data, feature, "max")[0])

            # Set up names of bins for graph display (relation to bin value)
            self.names[feature] = ["{} <= {}".format(feature, round(qts[0], 2))]
            for i in range(n_bins-1):
                self.names[feature].append(
                    "{} < {} <= {}".format(
                        round(qts[i], 2), feature, round(qts[i+1], 2)))
            self.names[feature].append("{} > {}".format(
                feature, round(qts[n_bins - 1], 2)))

            # Set up bucketizers based on bins
            splits = [-float("inf")] + qts + [float("inf")]
            self.bucketizers[feature] = Bucketizer(
                splits=splits, inputCol=feature, outputCol="disc_"+feature)
            discretized = self.bucketizers[feature].transform(data)

            # Calculate the statistics for each bucket
            self.means[feature] = []
            self.stds[feature] =[]
            for x in range(n_bins + 1):
                selection = discretized.filter(col(feature) == x)
                if not selection.head():
                    mean = 0
                    std = 0
                else:
                    mean = self._calculate_column_statistic(
                        selection, feature, "mean")[0]
                    std = self._calculate_column_statistic(
                        selection, feature, "std")[0]
                std += 0.00000000001
                self.means[feature].append(mean)
                self.stds[feature].append(std)
            self.mins[feature] = [boundaries[0]] + qts
            self.maxs[feature] = qts + [boundaries[1]]

    @staticmethod
    def _calculate_column_statistic(data, col_names, statistic):
        """
        Helper function for populating the statistics required for discretizer.
        :param data: pyspark.sql.DataFrame containing the features
        :param col_names: string or list/tuple of strings with the column names
        for which to calculate the statistic
        :param statistic: The name of the statistic function. One of
        "max", "min", "avg", "mean", or "std".
        :return: Dictionary of feature names to statistics.
        """
        if isinstance(col_names, str):
            col_names = [col_names]
        else:
            col_names = list(col_names)
        stats = []
        stats_methods = {
            "max": data.groupBy().max(*col_names).collect()[0][:],
            "min": data.groupBy().min(*col_names).collect()[0][:],
            "avg": data.groupBy().avg(*col_names).collect()[0][:],
            "mean": data.groupBy().avg(*col_names).collect()[0][:],
            "std": data.groupBy().agg(
                *[stddev(c) for c in col_names]).collect()[0][:]
        }
        try:
            stats = stats_methods[statistic]
        except KeyError:
            raise ValueError("Invalid statistic passed. Use one of {}"
                             .format(list(stats_methods.keys())))
        return list(stats)

    @abstractmethod
    def bins(self, data, labels):
        """
        To be overridden
        Returns for each feature to discretize the boundaries
        that form each bin of the discretizer
        """
        raise NotImplementedError("Must override bins() method")


class QuartileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None,
                 random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        data.registerTempTable("data")
        for feature in self.to_discretize:
            qts = self.spark.sql(
                    """
                    select
                        approx_percentile({fname}, 0.25) as p25,
                        approx_percentile({fname}, 0.50) as p50,
                        approx_percentile({fname}, 0.75) as p75
                    from
                        data
                    """
                    .format(fname=feature))\
                .withColumn("percentiles", array("p25", "p50", "p75"))\
                .collect()[0]["percentiles"]
            bins.append(qts)
        self.spark.catalog.dropTempView("data")
        return bins


class DecileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None,
                 random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        data.registerTempTable("data")
        percentiles = ["p{}".format(p) for p in range(10, 100, 10)]
        for feature in self.to_discretize:
            qts = self.spark.sql(
                    """
                    select
                        approx_percentile({fname}, 0.10) as p10,
                        approx_percentile({fname}, 0.20) as p20,
                        approx_percentile({fname}, 0.30) as p30,
                        approx_percentile({fname}, 0.40) as p40,
                        approx_percentile({fname}, 0.50) as p50,
                        approx_percentile({fname}, 0.60) as p60,
                        approx_percentile({fname}, 0.70) as p70,
                        approx_percentile({fname}, 0.80) as p80,
                        approx_percentile({fname}, 0.90) as p90,
                    from
                        data
                    """
                    .format(fname=feature))\
                .withColumn("percentiles", array(*percentiles))\
                .collect()[0]["percentiles"]
            bins.append(qts)
        self.spark.catalog.dropTempView("data")
        return bins
