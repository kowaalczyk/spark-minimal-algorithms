from typing import Dict, List, Tuple, Any
import random

import pytest
from pyspark import SparkContext, RDD

from spark_minimal_algorithms.algorithm import Step, Algorithm
from spark_minimal_algorithms.examples.countifs import (
    TeraSortWithLabels,
    GetResultsByLabel,
    AggregateResultsByQuery,
    Countifs,
)

random.seed(42)


@pytest.mark.parametrize(
    "cls",
    [
        TeraSortWithLabels,
        GetResultsByLabel,
        AggregateResultsByQuery,
    ],
)
@pytest.mark.parametrize("n_partitions", [1])
def test_step_creation(cls, spark_context, n_partitions):
    instance = cls(spark_context, n_partitions)

    assert isinstance(instance, Step) or isinstance(instance, Algorithm)
    assert instance._n_partitions == n_partitions


@pytest.mark.parametrize("n_partitions", [1, 2])
def test_algorithm_creation(spark_context, n_partitions):
    instance = Countifs(spark_context, n_partitions)

    assert isinstance(instance, Algorithm)
    assert instance._n_partitions == n_partitions

    assert hasattr(instance, "assign_next_label")
    assert type(instance.assign_next_label) == TeraSortWithLabels
    assert instance.assign_next_label._n_partitions == n_partitions

    assert hasattr(instance, "get_results_by_label")
    assert type(instance.get_results_by_label) == GetResultsByLabel
    assert instance.get_results_by_label._n_partitions == n_partitions

    assert hasattr(instance, "aggregate_results_by_query")
    assert type(instance.aggregate_results_by_query) == AggregateResultsByQuery
    assert instance.aggregate_results_by_query._n_partitions == n_partitions


@pytest.mark.parametrize("n_partitions", [1, 2, 3, 4])
def test_tera_sort_label_assignment_1d(spark_context, n_partitions):
    rdd = spark_context.parallelize([
        ((), (1,), (0, 0)),  # D0: data at x = 1 with empty label
        ((), (1,), (1, 0)),  # Q0: query at x = 1 with empty label
        ((), (2,), (0, 1)),  # D1: data at x = 2 with empty label
        ((), (2,), (1, 1)),  # Q1: query at x = 2 with empty label
    ])
    # Q0, D1 is the only pair of results that matches the COUNTIF criteria
    expected_result = [
        (((), ''), (1, 0)),
        (((), ''), (0, 1)),
    ]

    algorithm = TeraSortWithLabels(spark_context, n_partitions)
    result = algorithm(rdd=rdd).collect()

    assert result == expected_result


@pytest.mark.parametrize("n_partitions", [1, 2, 3, 4])
def test_tera_sort_label_assignment_2d_round_1_case_1(spark_context, n_partitions):
    rdd = spark_context.parallelize([
        ((), (3, 6), (0, 0)),  # D0
        ((), (4, 2), (0, 1)),  # D1
        ((), (0, 5), (1, 0)),  # Q0
        ((), (7, 1), (1, 1)),  # Q1
    ])
    # after 1st dimension, for Q0 both D1 and D2 are feasible
    expected_result_1st_round = [
        (((), ''), (5,), (1, 0)),
        (((), '0'), (5,), (1, 0)),
        (((), '0'), (6,), (0, 0)),
        (((), ''), (2,), (0, 1)),
    ]

    algorithm = TeraSortWithLabels(spark_context, n_partitions)
    result = algorithm(rdd=rdd).collect()

    assert result == expected_result_1st_round


@pytest.mark.parametrize("n_partitions", [1, 2, 3, 4])
def test_tera_sort_label_assignment_2d_round_2_case_1(spark_context, n_partitions):
    rdd_after_1st_round = spark_context.parallelize([
        (((), ''), (5,), (1, 0)),  # Q0
        (((), '0'), (5,), (1, 0)),  # Q0
        (((), '0'), (6,), (0, 0)),  # D0
        (((), ''), (2,), (0, 1)),  # D1
    ])
    # after 1st dimension, for Q0 both D1 and D2 are feasible
    expected_result_2nd_round = [
        ((((), '0'), ''), (1, 0)),
        ((((), '0'), ''), (0, 0)),
    ]

    algorithm = TeraSortWithLabels(spark_context, n_partitions)
    result = algorithm(rdd=rdd_after_1st_round).collect()

    assert result == expected_result_2nd_round


@pytest.mark.parametrize("n_partitions", [1, 2, 3, 4])
def test_tera_sort_label_assignment_2d_round_1_case_2(spark_context, n_partitions):
    # 'data_points': [(103, 480), (105, 1771), (1178, 101), (1243, 107)],
    # 'query_points': [(100, 100), (102, 102), (104, 104), (106, 106)]
    rdd = spark_context.parallelize([
        ((), (1178, 101), (0, 2)),
        ((), (103, 480), (0, 0)),
        ((), (105, 1771), (0, 1)),
        ((), (1243, 107), (0, 3)),
        ((), (104, 104), (1, 2)),
        ((), (100, 100), (1, 0)),
        ((), (102, 102), (1, 1)),
        ((), (106, 106), (1, 3))
    ])
    # after 1st dimension, for Q0 both D1 and D2 are feasible
    expected_result_1st_round = [
        (((), ''), (100,), (1, 0)),
        (((), '0'), (100,), (1, 0)),
        (((), '00'), (100,), (1, 0)),
        (((), ''), (102,), (1, 1)),
        (((), '0'), (102,), (1, 1)),
        (((), '0'), (480,), (0, 0)),
        (((), ''), (104,), (1, 2)),
        (((), ''), (1771,), (0, 1)),
        (((), '1'), (106,), (1, 3)),
        (((), ''), (101,), (0, 2)),
        (((), '1'), (101,), (0, 2)),
        (((), ''), (107,), (0, 3)),
        (((), '1'), (107,), (0, 3)),
        (((), '11'), (107,), (0, 3)),
    ]

    algorithm = TeraSortWithLabels(spark_context, n_partitions)
    result = algorithm(rdd=rdd).collect()

    assert result == expected_result_1st_round


@pytest.mark.parametrize("n_partitions", [1, 2, 3, 4])
def test_tera_sort_label_assignment_2d_round_2_case_2(spark_context, n_partitions):
    rdd_after_1st_round = spark_context.parallelize([
        (((), ''), (100,), (1, 0)),
        (((), '0'), (100,), (1, 0)),
        (((), '00'), (100,), (1, 0)),
        (((), ''), (102,), (1, 1)),
        (((), '0'), (102,), (1, 1)),
        (((), '0'), (480,), (0, 0)),
        (((), ''), (104,), (1, 2)),
        (((), ''), (1771,), (0, 1)),
        (((), '1'), (106,), (1, 3)),
        (((), ''), (101,), (0, 2)),
        (((), '1'), (101,), (0, 2)),
        (((), ''), (107,), (0, 3)),
        (((), '1'), (107,), (0, 3)),
        (((), '11'), (107,), (0, 3)),
    ])
    # after 1st dimension, for Q0 both D1 and D2 are feasible
    expected_result_2nd_round = [
        ((((), ''), ''), (1, 0)),
        ((((), ''), '0'), (1, 0)),
        ((((), ''), '00'), (1, 0)),
        ((((), ''), '00'), (0, 2)),
        ((((), ''), ''), (1, 1)),
        ((((), ''), '01'), (1, 1)),
        ((((), ''), ''), (1, 2)),
        ((((), ''), ''), (0, 3)),
        ((((), ''), ''), (0, 1)),
        ((((), ''), '10'), (0, 1)),
        ((((), '0'), ''), (1, 0)),
        ((((), '0'), '0'), (1, 0)),
        ((((), '0'), ''), (1, 1)),
        ((((), '0'), ''), (0, 0)),
        ((((), '00'), ''), (1, 0)),
        ((((), '1'), ''), (1, 3)),
        ((((), '1'), ''), (0, 3)),
    ]

    algorithm = TeraSortWithLabels(spark_context, n_partitions)
    result = algorithm(rdd=rdd_after_1st_round).collect()

    assert result == expected_result_2nd_round


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


def prepare_test_case(
    spark_context: SparkContext, test_case: Dict[str, List[Any]]
) -> Tuple[RDD, RDD, List[Any]]:
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
    return data_rdd, query_rdd, test_case["expected_result"]


@pytest.mark.parametrize("n_partitions", [1, 2, 4])
@pytest.mark.parametrize("test_case", TESTS_1D)
def test_algorithm_execution_1d(spark_context, n_partitions, test_case):
    data_rdd, query_rdd, expected_result = prepare_test_case(spark_context, test_case)

    countifs = Countifs(spark_context, n_partitions)
    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=1).collect()

    assert len(result) == len(expected_result)
    assert result == expected_result


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
    data_rdd, query_rdd, expected_result = prepare_test_case(spark_context, test_case)

    countifs = Countifs(spark_context, n_partitions)
    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=2).collect()

    assert len(result) == len(expected_result)
    assert result == expected_result


TESTS_3D = [
    {
        "query_points": [(100, 100, 100), (102, 102, 102)],
        "data_points": [
            (2137, 103, 480),
            (105, 2137, 1771),
            (1178, 101, 2137),
            (2137, 1243, 107),
        ],
        "expected_result": [(0, 4), (1, 3)],
    }
]


@pytest.mark.parametrize("n_partitions", [1, 2, 4])
@pytest.mark.parametrize("test_case", TESTS_3D)
def test_algorithm_execution_3d(spark_context, n_partitions, test_case):
    data_rdd, query_rdd, expected_result = prepare_test_case(spark_context, test_case)

    countifs = Countifs(spark_context, n_partitions)
    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=3).collect()

    assert len(result) == len(expected_result)
    assert result == expected_result


def random_test_case(n_data_points: int, n_query_points: int, n_dim: int):
    min_coord_value = 100
    max_coord_value = 100 + 100 * (n_query_points + n_data_points)

    def make_query_point(i):
        return tuple(min_coord_value + 2 * i for _ in range(n_dim))

    query_points = [make_query_point(i) for i in range(n_query_points)]

    data_points_per_query_point = n_data_points // n_query_points
    data_points_rest = n_data_points - n_query_points * data_points_per_query_point

    def make_data_point(min_val, max_val, global_max_val):
        """
        One of the coords will be between (min_val, max_val),
        rest of the coords will be between (min_val, global_max_val)
        """
        random_coord = random.randint(0, n_dim - 1)
        coords = [random.randint(min_val, global_max_val) for _ in range(n_dim)]
        coords[random_coord] = random.randint(min_val, max_val)
        return tuple(coords)

    # start with random data points which are smaller than all query points
    data_points = [
        make_data_point(0, min_coord_value, max_coord_value)
        for _ in range(data_points_rest)
    ]
    for i in range(n_query_points):
        # add data point in L-shape, with all dimensions > query point dimensions
        data_points_for_query = [
            make_data_point(
                min_coord_value + 2 * i + 1,
                min_coord_value + 2 * i + 1,
                max_coord_value,
            )
            for _ in range(data_points_per_query_point)
        ]
        data_points += data_points_for_query

    random.shuffle(data_points)

    expected_result = [
        (i, data_points_per_query_point * (n_query_points - i))
        for i in range(n_query_points)
    ]
    assert expected_result[-1] == (n_query_points - 1, data_points_per_query_point)
    assert (
        len(data_points) == n_data_points
    ), f"got: {len(data_points)}, expected: {n_data_points}"

    return {
        "data_points": data_points,
        "query_points": query_points,
        "expected_result": expected_result,
    }


LONG_N_PARTITIONS = [1, 2, 3, 4, 8, 16]


RANDOM_TESTS_1D = [
    random_test_case(10, 10, 1),
    random_test_case(1_000, 10, 1),
    random_test_case(1_000, 100, 1),
    random_test_case(1_000, 1_000, 1),
    # random_test_case(100_000, 10, 1),
    # random_test_case(100_000, 100, 1),
    # random_test_case(100_000, 1_000, 1),
    # random_test_case(100_000, 10_000, 1),
]


@pytest.mark.long
@pytest.mark.parametrize("n_partitions", LONG_N_PARTITIONS)
@pytest.mark.parametrize("test_case", RANDOM_TESTS_1D)
def test_algorithm_performance_1d(spark_context, n_partitions, test_case):
    data_rdd, query_rdd, expected_result = prepare_test_case(spark_context, test_case)

    countifs = Countifs(spark_context, n_partitions)
    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=1).collect()

    assert len(result) == len(expected_result)
    assert result == expected_result


RANDOM_TESTS_2D = [
    random_test_case(10, 10, 2),
    random_test_case(1_000, 10, 2),
    random_test_case(1_000, 100, 2),
    random_test_case(1_000, 1_000, 2),
    # random_test_case(100_000, 10, 2),
    # random_test_case(100_000, 100, 2),
    # random_test_case(100_000, 1_000, 2),
    # random_test_case(100_000, 10_000, 2),
]


@pytest.mark.long
@pytest.mark.parametrize("n_partitions", LONG_N_PARTITIONS)
@pytest.mark.parametrize("test_case", RANDOM_TESTS_2D)
def test_algorithm_performance_2d(spark_context, n_partitions, test_case):
    data_rdd, query_rdd, expected_result = prepare_test_case(spark_context, test_case)

    countifs = Countifs(spark_context, n_partitions)
    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=2).collect()

    assert len(result) == len(expected_result)
    assert result == expected_result


RANDOM_TESTS_3D = [
    random_test_case(10, 10, 3),
    random_test_case(1_000, 10, 3),
    random_test_case(1_000, 100, 3),
    random_test_case(1_000, 1_000, 3),
    # random_test_case(100_000, 10, 3),
    # random_test_case(100_000, 100, 3),
    # random_test_case(100_000, 1_000, 3),
    # random_test_case(100_000, 10_000, 3),
]


@pytest.mark.long
@pytest.mark.parametrize("n_partitions", LONG_N_PARTITIONS)
@pytest.mark.parametrize("test_case", RANDOM_TESTS_3D)
def test_algorithm_performance_3d(spark_context, n_partitions, test_case):
    data_rdd, query_rdd, expected_result = prepare_test_case(spark_context, test_case)

    countifs = Countifs(spark_context, n_partitions)
    result = countifs(data_rdd=data_rdd, query_rdd=query_rdd, n_dim=3).collect()

    assert len(result) == len(expected_result)
    assert result == expected_result
