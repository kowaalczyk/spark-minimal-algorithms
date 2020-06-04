import findspark
findspark.init()

import pytest
from pyspark import SparkConf, SparkContext

from spark_minimal_algorithms.algorithm import Step, Algorithm
from spark_minimal_algorithms.examples.countifs import (
    CountifsInitialStep,
    CountifsNextStep,
    CountifsResultsForLabel,
    CountifsResultsForQuery,
    Countifs,
)


@pytest.mark.parametrize(
    "cls",
    [
        CountifsInitialStep,
        CountifsNextStep,
        CountifsResultsForLabel,
        CountifsResultsForQuery,
    ],
)
@pytest.mark.parametrize("n_partitions", [1])
def test_step_creation(cls, spark_context, n_partitions):
    instance = cls(spark_context, n_partitions)

    assert isinstance(instance, Step)


@pytest.mark.parametrize("n_partitions", [1])
def test_algorithm_creation(spark_context, n_partitions):
    instance = Countifs(spark_context, n_partitions)

    assert isinstance(instance, Algorithm)
