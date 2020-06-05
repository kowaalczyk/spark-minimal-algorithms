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


TESTS_1D = [
    {
        "query_points": [1, 4, 5, 6, 7, 8, 11, 13, 14, 17],
        "data_points": [2, 3, 9, 10, 12, 15, 16, 18, 19, 20],
        "expected_result": [
            (0, 10),
            (1, 8),
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (6, 6),
            (7, 5),
            (8, 5),
            (9, 3),
        ],
    },
    {
        "query_points": sorted([9, 2, 4, 10, 7]),
        "data_points": [1, 3, 5, 6, 8],
        "expected_result": [(0, 4), (1, 3), (2, 1), (3, 0), (4, 0)],
    },
    {
        "query_points": [1],
        "data_points": [2, 3, 4, 5, 6, 7],
        "expected_result": [(0, 6)],
    },
    {
        "query_points": [1, 4, 5, 6, 8, 10, 13, 15, 16, 18],
        "data_points": [2, 3, 7, 9, 11, 12, 14, 17, 19, 20],
        "expected_result": [
            (0, 10),
            (1, 8),
            (2, 8),
            (3, 8),
            (4, 7),
            (5, 6),
            (6, 4),
            (7, 3),
            (8, 3),
            (9, 2),
        ],
    },
]


@pytest.mark.parametrize("n_partitions", [1, 2, 4])
@pytest.mark.parametrize("test_case", TESTS_1D)
def test_algorithm_execution_1d(spark_context, n_partitions, test_case):
    data_rdd = spark_context.parallelize(
        enumerate(
            map(lambda p: p if isinstance(p, tuple) else (p,), test_case["data_points"])
        )
    )
    query_rdd = spark_context.parallelize(
        enumerate(
            map(
                lambda p: p if isinstance(p, tuple) else (p,),
                sorted(test_case["query_points"]),
            )
        )
    )
    countifs = Countifs(spark_context, n_partitions)

    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=1).collect()

    assert result == test_case["expected_result"]


TESTS_2D = [
    {
        "data_points": [(3, 6), (4, 2)],
        "query_points": [(0, 5), (7, 1)],
        "expected_result": [(0, 1), (1, 0)],
    },
    {
        "query_points": [(0, 5), (7, 1)],
        "data_points": [(3, 6), (4, 2)],
        "expected_result": [(0, 1), (1, 0)],
    },
    {
        "query_points": [(100, 100), (102, 102)],
        "data_points": [(103, 480), (1178, 101)],
        "expected_result": [(0, 2), (1, 1)],
    },
    {
        "query_points": [(100, 100), (102, 102), (104, 104), (106, 106)],
        "data_points": [(1178, 101), (103, 480), (105, 1771), (1243, 107)],
        "expected_result": [(0, 4), (1, 3), (2, 2), (3, 1)],
    },
    {
        "query_points": [(100, 100), (102, 102)],
        "data_points": [(103, 480), (105, 1771), (1178, 101), (1243, 107)],
        "expected_result": [(0, 4), (1, 3)],
    },
]


@pytest.mark.parametrize("n_partitions", [1, 2, 4])
@pytest.mark.parametrize("test_case", TESTS_2D)
def test_algorithm_execution_2d(spark_context, n_partitions, test_case):
    data_rdd = spark_context.parallelize(
        enumerate(
            map(lambda p: p if isinstance(p, tuple) else (p,), test_case["data_points"])
        )
    )
    query_rdd = spark_context.parallelize(
        enumerate(
            map(
                lambda p: p if isinstance(p, tuple) else (p,),
                sorted(test_case["query_points"]),
            )
        )
    )
    countifs = Countifs(spark_context, n_partitions)

    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=2).collect()

    assert result == test_case["expected_result"]
