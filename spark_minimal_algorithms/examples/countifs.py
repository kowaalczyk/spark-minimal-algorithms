from typing import Dict, Any, Iterable, Optional, List, Tuple, Union
import math

from pyspark import RDD, Broadcast

from spark_minimal_algorithms.examples.tera_sort import SampleAndAssignBuckets
from spark_minimal_algorithms.algorithm import Step, Algorithm


# NOTE: IntEnum cannot be used, as unpickling would require entire project to be installed on compute nodes
DATA = 0
QUERY = 1


def _get_format_str(n_elements: int) -> str:
    binary_n_elements = int(math.ceil(math.log2(n_elements)))
    binary_format_str = "{" + ":0{}b".format(binary_n_elements) + "}"
    return binary_format_str


def _label_first_coord_and_type(point: Any) -> Any:
    label, coords, type_info = point
    return label, coords[0], type_info[0]


class SortAndAssignLabels(Step):
    """
    Replaces 2nd iteration of TeraSort to assign labels (on top of sorting the input).

    IN: (bucket index, (label(s), point coords, point type info))
    OUT: (label(s) + new label, point coords without first coord, point type)
        or just (labels, point type) if there are no more coords
    """

    @staticmethod
    def _sort_within_partition(bucket_and_points: Tuple[int, Iterable[Any]]) -> Tuple[int, Iterable[Any]]:
        bucket, points = bucket_and_points
        points = sorted(points, key=_label_first_coord_and_type)
        return bucket, points

    @staticmethod
    def group(rdd: RDD) -> RDD:  # type: ignore
        rdd = rdd.groupByKey().sortByKey()
        rdd = rdd.map(SortAndAssignLabels._sort_within_partition, preservesPartitioning=True)
        # for k, v in rdd.collect():  # todo: remove debug
        #     print(f"{k}: {list(v)}")
        return rdd

    @staticmethod
    def emit_by_group(group_key: int, group_items: Iterable[Any]) -> Optional[Any]:  # type: ignore
        bucket_idx = group_key

        first_label: Optional[str] = None
        n_points_for_first_label: Optional[int] = None
        last_label: Optional[str] = None
        n_points_for_last_label: Optional[int] = None
        for point in group_items:
            label, coords, type_info = point

            if first_label is None:
                first_label = label
                n_points_for_first_label = 1
            elif first_label == label:
                n_points_for_first_label += 1  # noqa: T484

            if last_label == label:
                n_points_for_last_label += 1  # noqa: T484
            else:
                last_label = label
                n_points_for_last_label = 1

        return bucket_idx, (first_label, n_points_for_first_label), (last_label, n_points_for_last_label)

    @staticmethod
    def broadcast(emitted_items: List[List[Any]]) -> Dict[str, Any]:  # type: ignore
        bucket_label_counts = sorted(emitted_items, key=lambda bucket_count: bucket_count[0])

        # print(f"bucket_label_counts: {bucket_label_counts}")  # todo: remove debug

        previous_label = ()  # empty tuple is never assigned as a label
        previous_count = 0
        bucket_prefix_counts = dict()  # i => (last label in (i-1)-th bucket, count of points with this label in previous buckets)
        total_label_counts = dict()  # label => total count of points with this label (only for multi-bucket labels)
        for bucket_count in bucket_label_counts:
            bucket_partition_idx = bucket_count[0]
            bucket_prefix_counts[bucket_partition_idx] = (previous_label, previous_count)

            first_label, first_label_count = bucket_count[1]
            last_label, last_label_count = bucket_count[2]

            if last_label == previous_label:
                # entire bucket consists of point with that one label
                previous_count += last_label_count
            else:
                # current bucket ends with different label than previous bucket
                total_label_counts[previous_label] = previous_count
                if first_label == previous_label:
                    # last label ends inside current bucket so we need to increase its count
                    total_label_counts[previous_label] += first_label_count

                previous_label = last_label
                previous_count = last_label_count

        # after iteration ends, we still need to assign total count for last label
        total_label_counts[previous_label] = previous_count

        keys_to_delete = {k for k in total_label_counts if total_label_counts[k] == 0}
        for k in keys_to_delete:
            del total_label_counts[k]

        # print(f"bucket_prefix_counts: {bucket_prefix_counts}")  # todo: remove debug
        # print(f"total_label_counts: {total_label_counts}")  # todo: remove debug

        return {
            "bucket_prefix_count": bucket_prefix_counts,
            "total_label_count": total_label_counts,
        }

    @staticmethod
    def step(  # type: ignore
        group_key: int, group_items: Iterable[Any], broadcast: Broadcast
    ) -> Iterable[Any]:
        bucket_idx = group_key
        prefix_counts: List[Tuple[str, int]] = broadcast.value["bucket_prefix_count"]
        bucket_prefix_count: Tuple[str, int] = prefix_counts[bucket_idx]
        previous_label, prefix_count_for_previous_label = bucket_prefix_count

        # get number of points for labels which span beyond current partition
        global_label_count: Dict[str, int] = broadcast.value["total_label_count"]
        global_labels = set(global_label_count.keys())

        # calculate number of points for each label (locally)
        local_label_count: Dict[str, int] = dict()
        for point in group_items:
            label, _, _ = point
            if label not in global_labels:
                try:
                    local_label_count[label] += 1
                except KeyError:
                    if label == previous_label:
                        local_label_count[label] = 1 + prefix_count_for_previous_label
                    else:
                        local_label_count[label] = 1

        # todo: label format strings for global labels can be pre-computed before broadcast
        # todo: we can probably get rid of few intermediate dicts to save memory
        label_count: Dict[str, int] = {
            **global_label_count,
            **local_label_count
        }
        # print(f"Caclulating label in partition: {group_key}")
        # print(f"available label counts: {label_count}")
        # print("")
        label_format_str = {
            label: _get_format_str(n_points_for_label)
            for label, n_points_for_label in label_count.items()
        }

        # assign new labels to points, based on combined counts of points per old label
        point_idx_within_label = prefix_count_for_previous_label
        for idx, point in enumerate(group_items):
            old_label, coords, type_info = point
            t, _ = type_info

            if old_label == previous_label:
                new_label = label_format_str[old_label].format(point_idx_within_label)
                point_idx_within_label += 1
            else:
                new_label = label_format_str[old_label].format(0)
                point_idx_within_label = 1
                previous_label = old_label

            # print(f"Point {point} (#{idx} in bucket #{bucket_idx}) got label {new_label}")  # todo: remove debug

            if t == DATA:
                for prefix_len in range(len(new_label)):
                    if new_label[prefix_len] == "1":
                        if len(coords) > 1:
                            yield (old_label, new_label[:prefix_len]), coords[1:], type_info
                        else:
                            yield (old_label, new_label[:prefix_len]), type_info

            elif t == QUERY:
                for prefix_len in range(len(new_label)):
                    if new_label[prefix_len] == "0":
                        if len(coords) > 1:
                            yield (old_label, new_label[:prefix_len]), coords[1:], type_info
                        else:
                            yield (old_label, new_label[:prefix_len]), type_info


class TeraSortWithLabels(Algorithm):
    __steps__ = {
        "assign_buckets": SampleAndAssignBuckets,
        "sort_and_assign_labels": SortAndAssignLabels,
    }

    def run(self, rdd: RDD) -> RDD:  # type: ignore
        rdd = rdd.cache()

        n_points = rdd.count()
        m = n_points / self.n_partitions
        optimal_p = math.log(n_points * self.n_partitions) / m

        rdd = self.assign_buckets(  # type: ignore
            rdd, p=optimal_p, key_func=_label_first_coord_and_type
        )
        rdd = self.sort_and_assign_labels(rdd)  # type: ignore

        return rdd


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
        "assign_next_label": TeraSortWithLabels,
        "get_results_by_label": GetResultsByLabel,
        "aggregate_results_by_query": AggregateResultsByQuery,
    }

    def run(self, data_rdd: RDD, query_rdd: RDD, n_dim: int) -> RDD:  # type: ignore
        empty_result_rdd = query_rdd.map(lambda idx_coords: (idx_coords[0], 0))

        data_rdd = data_rdd.map(
            lambda idx_coords: ((), idx_coords[1], (DATA, idx_coords[0]))
        )
        query_rdd = query_rdd.map(
            lambda idx_coords: ((), idx_coords[1], (QUERY, idx_coords[0]))
        )
        rdd = data_rdd.union(query_rdd)

        for _ in range(n_dim):
            rdd = self.assign_next_label(rdd=rdd)  # type: ignore

        rdd = empty_result_rdd.union(self.get_results_by_label(rdd))  # type: ignore
        rdd = self.aggregate_results_by_query(rdd).sortByKey()  # type: ignore
        return rdd
