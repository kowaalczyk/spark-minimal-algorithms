from typing import List, Tuple, Any
import random

import pytest

from spark_minimal_algorithms.examples.tera_sort import TeraSort

random.seed(42)


def create_test_case(n_points: int, n_dim: int) -> List[Tuple[Any]]:
    max_point = 100 * n_points
    points = [
        tuple(random.randint(1, max_point) for _ in range(n_dim))
        for _ in range(n_points)
    ]
    return points, sorted(points)


TESTS = [
    create_test_case(5, 1),
    create_test_case(10, 1),
    create_test_case(100, 1),
    create_test_case(5, 2),
    create_test_case(10, 2),
    create_test_case(100, 2),
    create_test_case(5, 3),
    create_test_case(10, 3),
    create_test_case(100, 3),
]


@pytest.mark.parametrize("test_case", TESTS)
@pytest.mark.parametrize("n_partitions", [1, 2, 4])
def test_tera_sort(spark_context, n_partitions, test_case):
    points, sorted_points = test_case
    n_dim = len(points[0])
    rdd = spark_context.parallelize(points)

    tera_sort = TeraSort(spark_context, n_partitions)
    result = tera_sort(rdd=rdd, n_dim=n_dim).collect()

    assert len(result) == len(sorted_points)
    assert result == sorted_points


LONG_TESTS = [
    create_test_case(100, 1),
    create_test_case(1_000, 1),
    create_test_case(100, 2),
    create_test_case(1_000, 2),
    create_test_case(100, 3),
    create_test_case(1_000, 3),
]


@pytest.mark.long
@pytest.mark.parametrize("test_case", LONG_TESTS)
@pytest.mark.parametrize("n_partitions", [1, 2, 3, 4, 8, 16])
def test_tera_sort_performance(spark_context, n_partitions, test_case):
    points, sorted_points = test_case
    n_dim = len(points[0])
    rdd = spark_context.parallelize(points)

    tera_sort = TeraSort(spark_context, n_partitions)
    result = tera_sort(rdd=rdd, n_dim=n_dim).collect()

    assert len(result) == len(sorted_points)
    assert result == sorted_points
