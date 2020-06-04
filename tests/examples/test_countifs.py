import findspark
findspark.init()

import pytest

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
    assert instance._n_partitions == n_partitions


@pytest.mark.parametrize("n_partitions", [1])
def test_algorithm_creation(spark_context, n_partitions):
    instance = Countifs(spark_context, n_partitions)

    assert isinstance(instance, Algorithm)
    assert instance._n_partitions == n_partitions

    assert hasattr(instance, "first_step")
    assert type(instance.first_step) == CountifsInitialStep
    assert instance.first_step._n_partitions == n_partitions

    assert hasattr(instance, "next_step")
    assert type(instance.next_step) == CountifsNextStep
    assert instance.next_step._n_partitions == n_partitions

    assert hasattr(instance, "results_for_label")
    assert type(instance.results_for_label) == CountifsResultsForLabel
    assert instance.results_for_label._n_partitions == n_partitions

    assert hasattr(instance, "results_for_query")
    assert type(instance.results_for_query) == CountifsResultsForQuery
    assert instance.results_for_query._n_partitions == n_partitions
