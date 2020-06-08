from typing import Iterable, Tuple, Any, List, Callable
from bisect import bisect_left
import random
import math

from spark_minimal_algorithms.algorithm import Step, Algorithm

from pyspark import RDD, Broadcast


class SampleAndAssignBuckets(Step):
    """
    IN: point coords
    OUT: (point bucket index, point coords), where
        buckets are created from randomly sampled points, and are ordered
        (bucket with higher index contains elements with higher values)
    """

    p = 0.1
    """ Default value for probability of sampling a point to be a bucket key """

    @staticmethod
    def extract_idx(
        partition_idx: int, partition_points: Iterable[Any]
    ) -> Iterable[Tuple[int, Any]]:
        for point in partition_points:
            yield partition_idx, point

    @staticmethod
    def group(rdd: RDD, **kwargs: Any) -> RDD:
        rdd = rdd.mapPartitionsWithIndex(
            SampleAndAssignBuckets.extract_idx, preservesPartitioning=True
        )
        rdd = rdd.groupByKey()
        return rdd

    @staticmethod
    def emit_by_group(group_key: int, group_items: Iterable[Any], **kwargs: Any) -> Any:
        samples = list()
        p: float = kwargs.get("p", SampleAndAssignBuckets.p)
        key_func: Callable[[Any], Any] = kwargs.get("key_func", lambda x: x)
        for point in group_items:
            if random.random() < p:
                sample_key = key_func(point)
                samples.append(sample_key)

        return samples

    @staticmethod
    def broadcast(emitted_items: List[List[Any]], **kwargs: Any) -> List[Any]:
        zero_point = ()  # empty tuple is always smaller than any n-dimensional point for n >= 1
        buckets = [zero_point] + [
            point for samples in emitted_items for point in samples
        ]
        return sorted(buckets)

    @staticmethod
    def step(  # type: ignore
        group_key: int, group_items: Iterable[Any], broadcast: Broadcast, **kwargs: Any
    ) -> Iterable[Tuple[int, Any]]:
        key_func: Callable[[Tuple[Any]], Tuple[Any]] = kwargs.get(
            "key_func", lambda x: x
        )
        for point in group_items:
            point_key = key_func(point)
            point_bucket = bisect_left(broadcast.value, point_key)
            yield point_bucket, point


class SortByKeyAndValue(Step):
    """
    IN: (point bucket index, point coords) in random order
    OUT: point coords in sorted order (by bucket index, and within a bucket by coords)
    """

    @staticmethod
    def group(rdd: RDD, **kwargs: Any) -> RDD:  # type: ignore
        rdd = rdd.groupByKey().sortByKey()
        return rdd

    @staticmethod
    def step(  # type: ignore
        group_key: int, group_items: Iterable[Any], broadcast: Broadcast, **kwargs: Any
    ) -> Iterable[Any]:
        key_func: Callable[[Tuple[Any]], Tuple[Any]] = kwargs.get(
            "key_func", lambda x: x
        )
        sorted_points = sorted(group_items, key=key_func)
        for point in sorted_points:
            yield point


class TeraSort(Algorithm):
    """
    Implements TeraSort - a minimal mapreduce algorithm for sorting data.

    Input:

        - `rdd`: RDD[point], where each point is a tuple of integers >= 1 (non-zero)

    Output:

        - `results_rdd`: sorted `rdd`

    """

    __steps__ = {
        "assign_buckets": SampleAndAssignBuckets,
        "sort": SortByKeyAndValue,
    }

    def run(self, rdd: RDD, key_func: Callable[[Tuple[Any]], Tuple[Any]] = lambda x: x) -> RDD:  # type: ignore
        rdd = rdd.cache()

        n_points = rdd.count()
        m = n_points / self.n_partitions
        optimal_p = math.log(n_points * self.n_partitions) / m

        rdd = self.assign_buckets(rdd, p=optimal_p, key_func=key_func)  # type: ignore
        rdd = self.sort(rdd, key_func=key_func)  # type: ignore

        return rdd
