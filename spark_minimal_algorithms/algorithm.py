from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Type, List, Tuple, Dict

from pyspark import SparkContext, RDD
from pyspark.broadcast import Broadcast


class Step(ABC):
    def __init__(self, sc: SparkContext, n_partitions: int):
        self._sc = sc
        self._n_partitions = n_partitions
        super().__init__()

    @staticmethod
    def group(rdd: RDD) -> RDD:
        return rdd.groupByKey()

    @staticmethod
    def emit_by_group(group_key: Any, group_items: Iterable[Any]) -> Optional[Any]:
        return None

    @staticmethod
    def broadcast(emitted_items: List[Any]) -> Optional[Any]:
        return None

    @abstractmethod
    def step(self, group_key: Any, group_items: Iterable[Any]) -> Iterable[Any]:
        pass

    def __call__(self, rdd: RDD) -> RDD:
        """
        Performs a single step of an algorithm.
        """
        if rdd.getNumPartitions() != self._n_partitions:
            rdd = rdd.repartition(self._n_partitions)

        step_cls: Type[Step] = self.__class__
        rdd = step_cls.group(
            rdd
        ).cache()  # cache because we use it twice (emit and step)

        def unwrap_emit(kv: Tuple[Any, Iterable[Any]]) -> Optional[Tuple[Any, Any]]:
            k, v = kv
            new_v = step_cls.emit_by_group(k, v)
            return new_v

        emitted = list(rdd.map(unwrap_emit).collect())
        to_broadcast = step_cls.broadcast(emitted)
        self.broadcast_: Broadcast = self._sc.broadcast(to_broadcast)

        def unwrap_step(kv: Tuple[Any, Iterable[Any]]) -> Iterable[Any]:
            k, v = kv
            for new_v in self.step(k, v):
                yield new_v

        rdd = rdd.flatMap(unwrap_step)
        return rdd


class Algorithm(ABC):
    __steps__: Dict[str, Type[Step]] = dict()

    def __init__(self, sc: SparkContext, n_partitions: int):
        self._n_partitions = n_partitions
        self._sc = sc

        for step_name, step_cls in self.__class__.__steps__.items():
            step_instance = step_cls(sc, n_partitions)
            setattr(self, step_name, step_instance)

        super().__init__()

    @abstractmethod
    def run(self, **kwargs: Dict[str, Any]) -> RDD:
        pass

    def __call__(self, **kwargs: Dict[str, Any]) -> RDD:
        for arg_name, arg in kwargs.items():
            if isinstance(arg, RDD) and arg.getNumPartitions() != self._n_partitions:
                kwargs[arg_name] = arg.repartition(self._n_partitions)

        rdd = self.run(**kwargs)
        return rdd


# iteration: gropuby, emit from group, aggregate and broadcast emitted values, map groups
# first round:
#   groupby: partition
#   emit: count per partition
#   aggregate and broadcast: prefix count per partition
#   map: partition -> labels for elements in partition
# next round:
#   groupby: label
#   emit: nothing
#   aggregate and broadcast: nothing
#   map: label group -> sort points, emit new labels
# summarize results 1:
#   groupby: label
#   emit: nothing
#   aggregate and broadcast: nothing
#   map: label group -> counts for query points in group
# summarize results 1:
#   groupby: query point
#   emit: nothing
#   aggregate and broadcast: nothing
#   map: sum values for query point
