from typing import Dict, Any, Iterable, Optional, List, Tuple, Union
import math

from pyspark import RDD, Broadcast

from spark_minimal_algorithms.algorithm import Step, Algorithm


# NOTE: IntEnum cannot be used, as unpickling would require entire project to be installed on compute nodes
DATA = 0
QUERY = 1


def _get_format_str(n_elements: int) -> str:
    binary_n_elements = int(math.ceil(math.log2(n_elements)))
    binary_format_str = "{" + ":0{}b".format(binary_n_elements) + "}"
    return binary_format_str


class SortAndAssignLabel(Step):
    """
    IN: (point coords, point type info) where:

        - point coords: (x1, x2, ...) have same number of dimensions for all points
        - point type info: (point type, index of point in the collection of points with this type)

    OUT: (label for the 1st dimension, point without 1st dimension)

        for each data point and each query point such that data point > query point
        at the first dimension

    """

    @staticmethod
    def select_key(coords_typeinfo: Any) -> Any:
        coords, type_info = coords_typeinfo
        t, order_for_t = type_info
        return (coords[0], t), (coords[1:], order_for_t)

    @staticmethod
    def unselect_key(selected_key_and_rest: Any) -> Any:
        selected_key, rest = selected_key_and_rest
        coord_0, t = selected_key
        other_coords, order_for_t = rest
        return (coord_0, other_coords, (t, order_for_t))

    @staticmethod
    def extract_partition_idx(
        idx: int, points: Iterable[Any]
    ) -> Iterable[Tuple[int, Any]]:
        for point in points:
            yield idx, point

    @staticmethod
    def group(rdd: RDD) -> RDD:  # type: ignore
        # sort by values - todo: consider using custom terasort implementation
        cls = SortAndAssignLabel
        rdd = rdd.map(cls.select_key).sortByKey().map(cls.unselect_key)
        rdd = rdd.mapPartitionsWithIndex(cls.extract_partition_idx).groupByKey()
        return rdd

    @staticmethod
    def emit_by_group(group_key: int, group_items: Iterable[Any]) -> Optional[Any]:  # type: ignore
        return group_key, len(list(group_items))

    @staticmethod
    def broadcast(  # type: ignore
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

    @staticmethod
    def step(  # type: ignore
        group_key: int, group_items: Iterable[Any], broadcast: Broadcast
    ) -> Iterable[Any]:
        prefix_counts: List[int] = broadcast.value["partition_prefix_count"]
        partition_prefix_count: int = prefix_counts[group_key]

        label_format_str: str = broadcast.value["label_format_str"]

        for idx, point in enumerate(group_items):
            coord_0, coords, type_info = point
            t, _ = type_info

            label = label_format_str.format(partition_prefix_count + idx)
            if t == DATA:
                for prefix_len in range(len(label)):
                    if label[prefix_len] == "1":
                        if len(coords) > 0:
                            yield label[:prefix_len], (coords, type_info)
                        else:
                            yield label[:prefix_len], type_info

            elif t == QUERY:
                for prefix_len in range(len(label)):
                    if label[prefix_len] == "0":
                        if len(coords) > 0:
                            yield label[:prefix_len], (coords, type_info)
                        else:
                            yield label[:prefix_len], type_info


class AssignNestedLabel(Step):
    """
    IN: (label, collection of points with label)

    OUT: (old label + new label for the 1st dimension, point without 1st dimension)
        for each data point and each query point such that data point > query point
        at the first dimension

    """

    @staticmethod
    def first_coord_and_point_type(point: Any) -> Any:
        coords, type_info = point
        return coords[0], type_info[0]

    @staticmethod
    def step(  # type: ignore
        group_key: Union[str, Tuple[str, ...]],
        group_items: Iterable[Any],
        broadcast: Broadcast,
    ) -> Iterable[Any]:
        points = sorted(group_items, key=AssignNestedLabel.first_coord_and_point_type)
        label_format_str = _get_format_str(len(points))
        old_label = group_key

        for idx, (coords, type_info) in enumerate(points):
            new_label = label_format_str.format(idx)

            t, _ = type_info
            if t == DATA:
                for prefix_len in range(len(new_label)):
                    if new_label[prefix_len] == "1":
                        if len(coords) > 1:
                            yield (old_label, new_label[:prefix_len]), (
                                coords[1:],
                                type_info,
                            )
                        else:
                            yield (old_label, new_label[:prefix_len]), type_info

            elif t == QUERY:
                for prefix_len in range(len(new_label)):
                    if new_label[prefix_len] == "0":
                        if len(coords) > 1:
                            yield (old_label, new_label[:prefix_len]), (
                                coords[1:],
                                type_info,
                            )
                        else:
                            yield (old_label, new_label[:prefix_len]), type_info


class GetResultsByLabel(Step):
    """
    IN: (label, points with this label)

    OUT: (query point idx, number of data points with label) for each query point with label

    """

    @staticmethod
    def step(  # type: ignore
        group_key: Union[str, Tuple[str, ...]],
        group_items: Iterable[Any],
        broadcast: Broadcast,
    ) -> Iterable[Any]:
        points = list(group_items)

        data_points = set(p[1] for p in points if p[0] == DATA)
        n_data_points = len(data_points)

        query_points = set(p[1] for p in points if p[0] == QUERY)
        for query_point_idx in query_points:
            yield query_point_idx, n_data_points


class AggregateResultsByQuery(Step):
    """
    IN: (query point index, collection of results for this query point for various labels)

    OUT: (query point index, total count of data points greater than this query point)

    """

    @staticmethod
    def step(group_key: int, group_items: Iterable[int], broadcast: Broadcast) -> Iterable[Tuple[int, int]]:  # type: ignore
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
        "sort_and_assign_label": SortAndAssignLabel,
        "assign_nested_label": AssignNestedLabel,
        "get_results_by_label": GetResultsByLabel,
        "aggregate_results_by_query": AggregateResultsByQuery,
    }

    def run(self, data_rdd: RDD, query_rdd: RDD, n_dim: int) -> RDD:  # type: ignore
        empty_result_rdd = query_rdd.map(lambda idx_coords: (idx_coords[0], 0))

        data_rdd = data_rdd.map(
            lambda idx_coords: (idx_coords[1], (DATA, idx_coords[0]))
        )
        query_rdd = query_rdd.map(
            lambda idx_coords: (idx_coords[1], (QUERY, idx_coords[0]))
        )
        rdd = data_rdd.union(query_rdd)

        rdd = self.sort_and_assign_label(rdd)  # type: ignore
        for _ in range(n_dim - 1):
            rdd = self.assign_nested_label(rdd)  # type: ignore

        rdd = empty_result_rdd.union(self.get_results_by_label(rdd))  # type: ignore
        rdd = self.aggregate_results_by_query(rdd).sortByKey()  # type: ignore
        return rdd
