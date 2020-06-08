from typing import List, Tuple, Any, Callable
import random

import pytest

from spark_minimal_algorithms.examples.tera_sort import TeraSort

random.seed(42)


def create_test_case(
    n_points: int,
    n_dim: int,
    key_func: Callable[[Tuple[Any]], Tuple[Any]] = lambda x: x,
) -> Tuple[List[Tuple[Any]], List[Tuple[Any]], Callable[[Tuple[Any]], Tuple[Any]]]:
    max_point = 100 * n_points
    points = [
        tuple(random.randint(1, max_point) for _ in range(n_dim))
        for _ in range(n_points)
    ]
    return points, sorted(points, key=key_func), key_func


TESTS = [
    create_test_case(5, 1),
    create_test_case(10, 1),
    create_test_case(10, 1, lambda x: (x[0],)),
    create_test_case(5, 2),
    create_test_case(10, 2),
    create_test_case(10, 2, lambda x: (x[0],)),
    create_test_case(5, 3),
    create_test_case(10, 3),
    create_test_case(10, 3, lambda x: (x[0],)),
]


@pytest.mark.parametrize("test_case", TESTS)
@pytest.mark.parametrize("n_partitions", [1, 2, 4])
def test_tera_sort(spark_context, n_partitions, test_case):
    points, sorted_points, key_func = test_case
    rdd = spark_context.parallelize(points)

    tera_sort = TeraSort(spark_context, n_partitions)
    result = tera_sort(rdd=rdd, key_func=key_func).collect()

    assert len(result) == len(sorted_points)
    assert result == sorted_points


LONG_TESTS = [
    create_test_case(100, 1),
    create_test_case(100, 1, lambda x: (x[0],)),
    create_test_case(1_000, 1),
    create_test_case(100, 2),
    create_test_case(100, 2, lambda x: (x[0],)),
    create_test_case(1_000, 2),
    create_test_case(100, 3),
    create_test_case(100, 3, lambda x: (x[0],)),
    create_test_case(1_000, 3),
]


@pytest.mark.long
@pytest.mark.parametrize("test_case", LONG_TESTS)
@pytest.mark.parametrize("n_partitions", [1, 2, 3, 4, 8, 16])
def test_tera_sort_performance(spark_context, n_partitions, test_case):
    points, sorted_points, key_func = test_case
    rdd = spark_context.parallelize(points)

    tera_sort = TeraSort(spark_context, n_partitions)
    result = tera_sort(rdd=rdd, key_func=key_func).collect()

    assert len(result) == len(sorted_points)
    assert result == sorted_points
