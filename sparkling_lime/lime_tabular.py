from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasFeaturesCol, \
    HasLabelCol, HasSeed, HasParallelism, HasPredictionCol, HasWeightCol
from pyspark.ml.param import Param, Params
from pyspark import keyword_only
from pyspark.ml.regression import LinearRegression
from multiprocessing.pool import ThreadPool
from pyspark.sql import DataFrame
import threading
from pyspark.sql.functions import lit, col
import pyspark.sql.functions as F
from sparkling_lime.sparkling_lime_base import BaseSparkMethods
from sparkling_lime.discretize import DecileDiscretizer, QuartileDiscretizer
from sparkling_lime.metrics import NeighborhoodGenerator, KernelWeight, PairwiseDistance
from pyspark.ml.feature import VectorAssembler
from math import sqrt
from sparkling_lime.utils import _parallelDatasetsTransformTasks, \
    _parallelDatasetsFitTasks, ReusedEstimator, ParallelTransformer
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler


class LocalLinearLearnerParams(Params):
    estimator = Param(Params._dummy(), "estimator",
                      "Regressor to use in explanation. Defaults to ridge"
                      " regression.")
    estimatorParamMap = Param(Params._dummy(), "estimatorParamMap", "")  # TODO

    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`
        """
        return self._set(estimator=value)

    def getEstimator(self):
        """
        Gets the value of estimator or its default value.
        """
        return self.getOrDefault(self.estimator)

    def setEstimatorParamMap(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMap`.
        """
        return self._set(estimatorParamMaps=value)

    def getEstimatorParamMap(self):
        """
        Gets the value of estimatorParamMap or its default value.
        """
        return self.getOrDefault(self.estimatorParamMap)


class LocalLinearLearner(HasFeaturesCol, HasLabelCol, HasSeed, HasWeightCol,
                         Estimator, ReusedEstimator, LocalLinearLearnerParams,
                         HasParallelism):
    """
    """

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMap=None,
                 featuresCol="neighborhood", labelCol="localLabel",
                 parallelism=1, seed=None):
        super().__init__()
        self._setDefault(featuresCol="neighborhood", labelCol="localLabel",
                         parallelism=1, estimatorParamMap=None,
                         estimator=LinearRegression(
                             regParam=1.0, elasticNetParam=0.0,
                             featuresCol=featuresCol, labelCol=labelCol,
                             weightCol="weight"))
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, estimator=None, estimatorParamMap=None,
                  labelCol="localLabel", featuresCol="neighborhood",
                  parallelism=1, seed=None):
        """
        Sets params for this KernelWeight transformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, datasets):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMap)
        parallelism = self.getOrDefault(self.parallelism)
        if isinstance(datasets, DataFrame):
            datasets = [datasets]
        else:
            datasets = list(datasets)
        numModels = len(datasets)
        pool = ThreadPool(processes=min(parallelism, numModels))
        models = [0] * len(datasets)
        tasks = _parallelDatasetsFitTasks(self, est, datasets, epm)
        for j, model in pool.imap_unordered(lambda f: f(), tasks):
            models[j] = model
            datasets[j].unpersist()
        llm = self._copyValues(LocalLinearLearnerModel(fittedModels=models,
                                                       parallelism=parallelism))
        return llm

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying estimator,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        newLLL = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newLLL.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMap remains the same
        return newLLL


class LocalLinearLearnerModel(HasFeaturesCol, HasLabelCol, HasPredictionCol,
                              HasSeed, LocalLinearLearnerParams,
                              Transformer, ParallelTransformer):
    """
    NOTE: Currently r2 score ignores weight column. Will be updated with
    future versions of pyspark.
    """
    @keyword_only
    def __init__(self, fittedModels=(), parallelism=1, scoreCol="r2",
                 coeffCol="coeff", interceptCol="intercept"):
        super().__init__()
        self.parallelism = parallelism
        self.scoreCol = scoreCol
        self.coeffCol = coeffCol
        self.interceptCol = interceptCol
        self.fittedModels = list(fittedModels)

    def _transform(self, datasets):
        models = self.fittedModels
        if isinstance(datasets, DataFrame):
            datasets = [datasets]
        else:
            datasets = list(datasets)
        numModels = len(models)

        if numModels != len(datasets):
            raise ValueError("Number of models did not match number of datasets"
                             " to transform.")
        pool = ThreadPool(processes=min(self.parallelism, numModels))
        transformed_datasets = [0] * len(datasets)
        tasks = _parallelDatasetsTransformTasks(self, models, datasets)
        for j, transformed_data in pool.imap_unordered(lambda f: f(), tasks):
            model = self.fittedModels[j]
            model_summary = model.summary
            transformed_datasets[j] = transformed_data\
                .withColumn(self.scoreCol, lit(model_summary.r2))\
                .withColumn(self.coeffCol,
                            F.array(*[lit(c) for c in model.coefficients]))\
                .withColumn(self.interceptCol, lit(model.intercept))
            datasets[j].unpersist()
        return transformed_datasets

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying estimator,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        fittedModels = [m.copy(extra) for m in self.fittedModels]
        parallelism = self.parallelism
        return LocalLinearLearnerModel(fittedModels, parallelism)


class LimeTabularExplainer(BaseSparkMethods):
    """
    Convenience class for calculating explanations without manually invoking
    all the necessary pipeline methods.
    """

    def __init__(self, predictions, featureNames, globalEstimator,
                 mode="classification",
                 categoricalFeatures=None, kernelWidth=None,
                 discretizeContinuous=True, discretizer="quartile",
                 sampleAroundInstance=False, seed=None):
        """
        :param predictions: `pyspark.sql.DataFrame` with predictions generated
        by `globalEstimator` and unassmbled features. Categorical features
        should already be indexed.
        :param featureNames: Iterable of feature names corresponding to columns
        in `predictions`.
        :param globalEstimator: A `pyspark.ml.Estimator` used to generate
        `predictions`
        :param mode: "classification" or "regression"; whether `globalEstimator`
        is a classifier or regressor.
        :param categoricalFeatures: Iterable of feature names corresponding to
        categorical features in the data.
        :param kernelWidth: Kernel width for the exponential kernel. If none,
        defaults to sqrt(#columns) * 0.75
        :param discretizeContinuous: if True, all non-categorical features will
        be discretized.
        :param discretizer: Type of discretizer ("quartile" or "decile"),
        only used if `discretizeContinuous` is True. Defaults to "quartile".
        :param sampleAroundInstance: If True, will sample continuous features
        in perturbed samples from a normal centered at the instance being
        explained. Otherwise, the normal is centered on the mean of the feature
        data.
        :param seed: Random seed
        """
        super().__init__()
        self.seed = seed
        self.predictions = predictions
        self.featureNames = list(featureNames)
        self.globalEstimator = globalEstimator
        self.mode = mode
        self.categoricalFeatures = list(categoricalFeatures)
        self.kernelWidth = kernelWidth if kernelWidth \
            else sqrt(len(featureNames)) * 0.75
        self.discretizeContinuous = discretizeContinuous
        self.sampleAroundInstance = sampleAroundInstance
        if discretizeContinuous:
            if discretizer == "quartile":
                self.discretizer = QuartileDiscretizer(self.predictions,
                                                       self.categoricalFeatures,
                                                       self.featureNames,
                                                       labels=None,
                                                       random_state=seed)
            elif discretizer == "decile":
                self.discretizer = DecileDiscretizer(self.predictions,
                                                     self.categoricalFeatures,
                                                     self.featureNames,
                                                     labels=None,
                                                     random_state=seed)
            else:
                raise ValueError("Invalid argument for `discretizer`: {}\n"
                                 "Please choose one of ['quartile', 'decile']"
                                 .format(discretizer))
        else:
            self.discretizer = None

    def explain_instances(self, path, globalTransformer, localEstimator=None,
                          idCol="explanationId", sampleSize=5000,
                          parallelism=6):
        """
        Requires data with predictions (labels) and unassembled features
        """
        kernelWidth = self.kernelWidth
        orig_cols = self.predictions.columns
        # Assemble features for training, and to keep after discretization
        origAssembler = VectorAssembler(inputCols=self.featureNames,
                                        outputCol="origFeatures")
        dataset = origAssembler.transform(self.predictions)
        if self.discretizer:
            dataset = self.discretizer.discretize(dataset)
        output_features = ["n_{}".format(c) for c in self.featureNames]

        # Add unique id for separation
        dataset = dataset.withColumn(
            idCol, F.monotonically_increasing_id())
        dataset.cache()
        ids = [r[0] for r in dataset.select(idCol).collect()]
        # Broadcast the list of ids (will generate explanation for each)
        b_ids = self.sc.broadcast(ids)

        # Generate neighborhood of peturbed samples
        neighborhoodGen = NeighborhoodGenerator(
            inputCols=self.featureNames,
            inverseOutputCols=output_features,
            neighborhoodSize=sampleSize,
            discretizer=self.discretizer,
            seed=self.seed,
            sampleAroundInstance=self.sampleAroundInstance,
            categoricalCols=self.categoricalFeatures)

        # Generate global predictions for neighborhood
        neighborAssembler = VectorAssembler(inputCols=output_features,
                                            outputCol="neighborhood")
        extra_params = ParamGridBuilder()\
            .addGrid(globalTransformer.featuresCol, ["neighborhood"])\
            .addGrid(globalTransformer.predictionCol, ["globalLabel"])\
            .addGrid(globalTransformer.probabilityCol, ["globalProbability"])\
            .addGrid(globalTransformer.rawPredictionCol, ["globalRawPrediction"])\
            .build()[0]
        # globalTransformer._set(featuresCol="neighborhood")\
        #     ._set(predictionCol="globalLabel")\
        #     ._set(probabilityCol="globalProbability")\
        #     ._set(rawPrediction="globalRawPrediction")

        predictPipe = Pipeline(stages=[neighborhoodGen, neighborAssembler,
                                       globalTransformer])
        neighborhood = predictPipe.fit(dataset)\
            .transform(dataset, extra_params)\
            .withColumn("featurePair", F.struct("origFeatures", "neighborhood"))

        # Calculate distance and kernel weight
        distanceCalculator = PairwiseDistance(inputCol="featurePair",
                                              outputCol="distance")
        kernelCalculator = KernelWeight(inputCol="distance",
                                        outputCol="weight",
                                        kernelWidth=kernelWidth)
        dist_neighborhood = distanceCalculator.transform(neighborhood)\
            .drop("featurePair")
        weight_neighborhood = kernelCalculator.transform(dist_neighborhood)\
            .drop("distance")

        # Standard scale the features prior to learning the linear models
        continuous_cols = ["inverse_"+c for c in self.featureNames
                           if c not in self.categoricalFeatures]
        scalers = [StandardScaler(inputCol=c, outputCol="scaled_"+c)
                                  for c in continuous_cols]
        scalePipe = Pipeline.stages(scalers)
        scaled_neighborhood = Pipeline.fit(weight_neighborhood)\
            .transform(weight_neighborhood)

        # Save each user into separate partitions
        weight_neighborhood.drop(*self.featureNames)\
            .write.partitionBy(idCol)\
            .mode("append")\
            .format("orc")\
            .save(path)
        neighborhood.unpersist()
        dataset.unpersist()
        # Load the neighborhood data back into array, one partition at a time
        neighborhoods = []
        for id in b_ids.value:
            hood = self.spark.read.orc("{}/{}={}".format(path, idCol, id))
            neighborhoods.append(hood)

        lll = LocalLinearLearner(estimator=localEstimator, parallelism=6,
                                 featuresCol="neighborhood")
        localModels = lll.fit(neighborhoods)
        explanations = localModels.transform(neighborhoods)
        for e in explanations:
            # All the columns we care about are duplicated through neighborhood,
            # so just aggregate and take the first
            e.groupBy(idCol).agg(F.first("origFeatures").alias("origFeatures"),
                                 F.first("r2").alias("r2"),
                                 F.first("coeff").alias("coeff"),
                                 F.first("intercept").alias("intercept"),
                                 *[F.first(c).alias(c) for c in orig_cols])
        return explanations

        # --- Load predictions w/ unassembled features ---
        # --- Initialize a discretizer ---
        # Scale the features
        # --- Generate neighborhood out of features ---
        # --- Save with partitionBy customerId (or other id col) ---
        #  --- Reload partitions separately into an array (should be a single row) ---
        # Calculate distances + kernel weight
        # --- Learn local models on each dataframe in array ---
        # ---  Generate transformed data, take single row with initial vector of features + linear model info ---
        # Save (/union?/make df out of single rows?)
        #



