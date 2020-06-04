from typing import Dict, Any, Iterable, Optional, List, Tuple, Union
import math
from enum import IntEnum

from pyspark import RDD

from spark_minimal_algorithms.algorithm import Step, Algorithm


class PointType(IntEnum):
    """
    Represents a type of point (query or data).
    IntEnum is used to ensure correct sorting order (data before query).
    """

    DATA = 0
    QUERY = 1


def _get_format_str(n_elements: int) -> str:
    binary_n_elements = int(math.ceil(math.log2(n_elements)))
    binary_format_str = "{" + ":0{}b".format(binary_n_elements) + "}"
    return binary_format_str


class CountifsInitialStep(Step):
    """
    IN: (point coords, point type info) where:

        - point coords: (x1, x2, ...) have same number of dimensions for all points
        - point type info: (point type, index of point in the collection of points with this type)

    OUT: (label for the 1st dimension, point without 1st dimension)

        for each data point and each query point such that data point > query point
        at the first dimension

    """

    @staticmethod
    def group(rdd: RDD) -> RDD:
        def select_key(coords_typeinfo: Any) -> Any:
            coords, type_info = coords_typeinfo
            t, order_for_t = type_info
            return (coords[0], t), (coords[1:], order_for_t)

        def unselect_key(selected_key_and_rest: Any) -> Any:
            selected_key, rest = selected_key_and_rest
            coord_0, t = selected_key
            other_coords, order_for_t = rest
            return (coord_0, other_coords, (t, order_for_t))

        # sort by values - todo: consider using custom terasort implementation
        rdd = rdd.map(select_key).sortByKey().map(unselect_key)

        def extract_partition_idx(
            idx: int, points: Iterable[Any]
        ) -> Iterable[Tuple[int, Any]]:
            for point in points:
                yield idx, points

        rdd = rdd.mapPartitionsWithIndex(extract_partition_idx)
        return rdd

    @staticmethod
    def emit_by_group(group_key: int, group_items: Iterable[Any]) -> Optional[Any]:
        return group_key, len(list(group_items))

    @staticmethod
    def broadcast(
        emitted_items: List[Tuple[int, int]]
    ) -> Dict[str, Union[str, List[int]]]:
        parition_counts = [
            idx_count[1]
            for idx_count in sorted(emitted_items, key=lambda idx_count: idx_count[0])
        ]
        partition_prefix_counts = [
            sum(parition_counts[:i]) for i in range(len(parition_counts))
        ]

        total_count = partition_prefix_counts[-1] + parition_counts[-1]
        label_format_str = _get_format_str(total_count)

        return {
            "partition_prefix_count": partition_prefix_counts,
            "label_format_str": label_format_str,
        }

    def step(self, group_key: int, group_items: Iterable[Any]) -> Iterable[Any]:
        prefix_counts: List[int] = self.broadcast_.value["partition_prefix_count"]
        partition_prefix_count: int = prefix_counts[group_key]

        label_format_str: str = self.broadcast_.value["label_format_str"]

        for idx, point in enumerate(group_items):
            coord_0, coords, type_info = point
            t, _ = type_info

            label = label_format_str.format(partition_prefix_count + idx)
            if t == PointType.DATA:
                for prefix_len in range(len(label)):
                    if label[prefix_len] == "1":
                        if len(coords) > 0:
                            yield label[:prefix_len], (coords, type_info)
                        else:
                            yield label[:prefix_len], type_info

            elif t == PointType.QUERY:
                for prefix_len in range(len(label)):
                    if label[prefix_len] == "0":
                        if len(coords) > 0:
                            yield label[:prefix_len], (coords, type_info)
                        else:
                            yield label[:prefix_len], type_info


class CountifsNextStep(Step):
    """
    IN: (label, collection of points with label)

    OUT: (old label + new label for the 1st dimension, point without 1st dimension)
        for each data point and each query point such that data point > query point
        at the first dimension

    """

    def step(
        self, group_key: Union[str, Tuple[str, ...]], group_items: Iterable[Any]
    ) -> Iterable[Any]:
        def first_coord_and_point_type(point: Any) -> Any:
            coords, type_info = point
            return coords[0], type_info[0]

        points = sorted(group_items, key=first_coord_and_point_type)
        label_format_str = _get_format_str(len(points))
        old_label = group_key

        for idx, (coords, type_info) in enumerate(points):
            new_label = label_format_str.format(idx)

            t, _ = type_info
            if t == PointType.DATA:
                for prefix_len in range(len(new_label)):
                    if new_label[prefix_len] == "1":
                        if len(coords) > 1:
                            yield (old_label, new_label[:prefix_len]), (
                                coords[1:],
                                type_info,
                            )
                        else:
                            yield (old_label, new_label[:prefix_len]), type_info

            elif t == PointType.QUERY:
                for prefix_len in range(len(new_label)):
                    if new_label[prefix_len] == "0":
                        if len(coords) > 1:
                            yield (old_label, new_label[:prefix_len]), (
                                coords[1:],
                                type_info,
                            )
                        else:
                            yield (old_label, new_label[:prefix_len]), type_info


class CountifsResultsForLabel(Step):
    """
    IN: (label, points with this label)

    OUT: (query point idx, number of data points with label) for each query point with label

    """

    def step(
        self, group_key: Union[str, Tuple[str, ...]], group_items: Iterable[Any]
    ) -> Iterable[Any]:
        points = list(group_items)

        data_points = set(p[1] for p in points if p[0] == PointType.DATA)
        n_data_points = len(data_points)

        query_points = set(p[1] for p in points if p[0] == PointType.QUERY)
        for query_point_idx in query_points:
            yield query_point_idx, n_data_points


class CountifsResultsForQuery(Step):
    """
    IN: (query point index, collection of results for this query point for various labels)

    OUT: (query point index, total count of data points greater than this query point)

    """

    def step(
        self, group_key: int, group_items: Iterable[int]
    ) -> Iterable[Tuple[int, int]]:
        yield group_key, sum(group_items)


class Countifs(Algorithm):
    """
    Implements the Multidimensional Interval Multiquery Processor algorithm, that for each
    point in `query_rdd` counts number of points in `query_rdd` greater than this point
    (at every dimension).

    Input:

        - `query_rdd`: RDD[query point index, query point coords]
        - `data_rdd`: RDD[data point index, data point coords]
        - `n_dim`: int - number of dimensions (coordinates)

    Output:

        - `results_rdd`: RDD[query point index, number of data points greater than this query point]
    """

    __steps__ = {
        "first_step": CountifsInitialStep,
        "next_step": CountifsNextStep,
        "results_for_label": CountifsResultsForLabel,
        "results_for_query": CountifsResultsForQuery,
    }

    def run(self, data_rdd: RDD, query_rdd: RDD, n_dim: int) -> RDD:
        data_rdd = data_rdd.map(
            lambda idx_coords: (idx_coords[1], (PointType.DATA, idx_coords[0]))
        )
        query_rdd = query_rdd.map(
            lambda idx_coords: (idx_coords[1], (PointType.QUERY, idx_coords[0]))
        )
        rdd = data_rdd.union(query_rdd)

        rdd = self.first_step(rdd)
        for _ in range(n_dim - 1):
            rdd = self.next_step(rdd)

        rdd = self.results_for_label(rdd)
        rdd = self.results_for_query(rdd)
        return rdd
