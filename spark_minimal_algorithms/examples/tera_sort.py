from typing import Iterable, Tuple, Any, List
from bisect import bisect_left
import random
import math

from spark_minimal_algorithms.algorithm import Step, Algorithm

from pyspark import RDD, Broadcast


class TeraSortFirstRound(Step):
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
        rdd = rdd.mapPartitionsWithIndex(TeraSortFirstRound.extract_idx).groupByKey()
        return rdd

    @staticmethod
    def emit_by_group(group_key: int, group_items: Iterable[Any], **kwargs: Any) -> Any:
        samples = list()
        p: float = kwargs.get("p", TeraSortFirstRound.p)
        for point in group_items:
            if random.random() < p:
                samples.append(point)

        return samples

    @staticmethod
    def broadcast(emitted_items: List[List[Any]], **kwargs: Any) -> List[Any]:
        n_dim = kwargs["n_dim"]
        zero_point = tuple(0 for _ in range(n_dim))
        buckets = [zero_point] + [
            point for samples in emitted_items for point in samples
        ]
        return sorted(buckets)

    @staticmethod
    def step(  # type: ignore
        group_key: int, group_items: Iterable[Any], broadcast: Broadcast, **kwargs: Any
    ) -> Iterable[Tuple[int, Any]]:
        for point in group_items:
            point_bucket = bisect_left(broadcast.value, point)
            yield point_bucket, point


class TeraSortFinalRound(Step):
    @staticmethod
    def group(rdd: RDD) -> RDD:  # type: ignore
        rdd = rdd.groupByKey().sortByKey()
        return rdd

    @staticmethod
    def step(  # type: ignore
        group_key: int, group_items: Iterable[Any], broadcast: Broadcast
    ) -> Iterable[Any]:
        sorted_points = sorted(group_items)
        for point in sorted_points:
            yield point


class TeraSort(Algorithm):
    __steps__ = {
        "assign_buckets": TeraSortFirstRound,
        "sort": TeraSortFinalRound,
    }

    def run(self, rdd: RDD, n_dim: int) -> RDD:  # type: ignore
        rdd = rdd.cache()

        n_points = rdd.count()
        m = n_points / self.n_partitions
        optimal_p = math.log(n_points * self.n_partitions) / m

        rdd = self.assign_buckets(rdd, p=optimal_p, n_dim=n_dim)  # type: ignore
        rdd = self.sort(rdd)  # type: ignore

        return rdd
