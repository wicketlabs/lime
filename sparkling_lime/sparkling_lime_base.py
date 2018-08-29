from pyspark.sql import SparkSession


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