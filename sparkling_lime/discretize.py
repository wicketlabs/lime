"""
Discretizers classes, to be used in lime_tabular
"""
import numpy as np
from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
from pyspark.sql.functions import col, stddev, randn, when
from pyspark.sql import DataFrameStatFunctions as stats
from pyspark.ml.feature import Bucketizer
from pyspark.ml import Pipeline
from sparkling_lime.sparkling_lime_base import BaseSparkMethods


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
        :param data: pyspark.sql.DataFrame of unassembled features
        :param categorical_features: list of names (str) corresponding to the
        categorical columns. These features will not be discretized.
        Everything else will be considered continuous, and will be discretized.
        :param feature_names: list of names (strings) corresponding to the
        columns in the training data.
        """
        super().__init__()
        self.to_discretize = ([x for x in feature_names
                              if x not in categorical_feature_names])
        self.names = {}
        self.bucketizers = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.feature_names = feature_names
        self.random_state = check_random_state(random_state)
        self.categorical_features_names = categorical_feature_names

        bins = self.bins(data, labels)
        bins = [np.unique(x).tolist() for x in bins]

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = len(qts)   # Number of borders (=#bins-1)
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
            self.stds[feature] = []
            for x in range(n_bins + 1):
                selection = discretized.filter(col("disc_"+feature) == x)
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

    def discretize(self, data):
        """
        Discretizes the data.
        :param data: pyspark.sql.DataFrame
        :return: a pyspark.sql.DataFrame with same dimension, but discretized
        """
        other_cols = [c for c in data.columns if c not in self.feature_names]
        pipe = Pipeline(stages=list(self.bucketizers.values()))
        discretized = pipe.fit(data).transform(data)
        disc_select = [col(c).alias(c.lstrip("disc_"))
                       for c in discretized.columns if c.startswith("disc_")]
        discretized = discretized.select(
            *(disc_select + self.categorical_features_names + other_cols))
        return discretized

    def undiscretize(self, data):
        return self.static_undiscretize(data, self.to_discretize,
                                         self.means, self.stds, self.mins,
                                         self.maxs, self.random_state)
    @staticmethod
    def static_undiscretize(data, subset, means, stds, mins, maxs, random_state):
        for feature in subset:
            means_replacement = dict([(idx, val) for idx, val
                                      in enumerate(means[feature])])
            std_replacement = dict([(idx, val) for idx, val
                                    in enumerate(stds[feature])])
            max_replacement = dict([(idx, val) for idx, val
                                    in enumerate(maxs[feature])])
            min_replacement = dict([(idx, val) for idx, val
                                    in enumerate(mins[feature])])
            # Add dummy columns to reference the means, etc. for feature buckets
            data = data.withColumn("mean_"+feature, col(feature))\
                .withColumn("std_"+feature, col(feature))\
                .withColumn("max_"+feature, col(feature))\
                .withColumn("min_"+feature, col(feature))
            # Populate the values according to stored information
            data = data.replace(means_replacement, subset="mean_"+feature)
            data = data.replace(std_replacement, subset="std_"+feature)
            data = data.replace(max_replacement, subset="max_"+feature)
            data = data.replace(min_replacement, subset="min_"+feature)
            data = data.withColumn(
                    "rand_"+feature,
                    randn(seed=random_state.randint(-100, 100)))\
                .withColumn(
                    feature,
                    col("rand_"+feature)*col("std_"+feature)+col("mean_"+feature))
            data = data.withColumn(
                feature,
                # Bucketizer is from [x, y)
                when(col(feature) >= col("max_"+feature),
                     col("max_"+feature)-.000000001)\
                .when(col(feature) < col("min_"+feature), col("min_"+feature))\
                .otherwise(col(feature)))
            data = data.drop("rand_"+feature, "min_"+feature, "max_"+feature,
                             "std_"+feature, "mean_"+feature)
        return data


class QuartileDiscretizer(BaseDiscretizer):
    """
    Bins continuous data into quartiles.
    """
    def __init__(self, data, categorical_features, feature_names, labels=None,
                 random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = stats(data)\
            .approxQuantile(self.to_discretize, [0.25, 0.50, 0.75], 0.001)
        return bins


class DecileDiscretizer(BaseDiscretizer):
    """
    Bins continuous data into deciles.
    """
    def __init__(self, data, categorical_features, feature_names, labels=None,
                 random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        percentiles = [p/100.0 for p in range(10, 100, 10)]
        bins = stats(data)\
            .approxQuantile(self.to_discretize, percentiles, 0.001)
        return bins
