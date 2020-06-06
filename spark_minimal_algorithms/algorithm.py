from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Any, Optional, Iterable, Type, List, Tuple, Dict

from pyspark import SparkContext, RDD
from pyspark.broadcast import Broadcast


class Step(ABC):
    """
    `Step` (sometimes called a round) represents the following sequence of operations:
    1. Grouping of items
    2. Emitting some data from each group
    3. Combining emitted data and broadcasting it to all groups
    4. Transforming each group (using it's original items and the broadcasted values)

    Operations 1-3 are optional, and the framework provides reasonable defaults.
    The life-cycle of a step matches these operations directly:
    1. `Step.group` is called to perform the grouping (by default, performs `pyspark.RDD.groupByKey`)
    2. `Step.emit_by_group` is called on each group to emit some values (by default, nothing is emitted)
    3. `Step.broadcast` is called on a list of emitted items and its return value is broadcasted to all groups
    (by default, nothing is broadcasted)
    4. `Step.step` is called on each group, and can access broadcasted value to transform the group

    To implement a new Step (to be used in some new algorithm), you can subclass `Step` and implement its methods.

    To create an instance of `Step`, you need to provide it with spark context and desired number of partitions.
    Calling instance of a `Step` class ensures all of these operations are performed in correct order, for example:
    ```python
    from spark_minimal_algorithms.examples.tera_sort import SampleAndAssignBuckets

    # prepare some dummy data in compatible format:
    input_rdd = spark_context.parallelize((i, i) for i in range(10, 0, -1))

    # SampleAndAssignBuckets is a subclass of Step:
    step = SampleAndAssignBuckets(spark_context, n_partitions)

    # call the step to perform its operations in correct order:
    output_rdd = step(input_rdd)

    # result should be sorted in ascending order:
    print(input_rdd.collect())
    ```
    """

    def __init__(self, sc: SparkContext, n_partitions: int):
        """
        Initializes instance of the step.

        **DO NOT OVERRIDE WHEN DEFINING CUSTOM STEPS.**
        """
        self._sc = sc
        self._n_partitions = n_partitions
        super().__init__()

    @staticmethod
    def group(rdd: RDD, **kwargs: Any) -> RDD:
        """
        Performs grouping stage of the step.

        **Must to return (key, value) pairs.**

        Optional kwargs contain anything that was passed when calling the algorithm.
        """
        return rdd.groupByKey()

    @staticmethod
    def emit_by_group(
        group_key: Any, group_items: Iterable[Any], **kwargs: Any
    ) -> Optional[Any]:
        """
        Called on each group (result of `Step.group`), can be used to emit
        and later broadcast arbitrary values from one group to others.

        Optional kwargs contain anything that was passed when calling the algorithm.
        """
        return None

    @staticmethod
    def broadcast(emitted_items: List[Any], **kwargs: Any) -> Optional[Any]:
        """
        Called on a list of items emitted using `Step.emit_by_group`, anything
        that is returned by this function is broadcasted to all groups.

        Optional kwargs contain anything that was passed when calling the algorithm.
        """
        return None

    @abstractstaticmethod
    def step(
        group_key: Any, group_items: Iterable[Any], broadcast: Broadcast, **kwargs: Any
    ) -> Iterable[Any]:
        """
        Called on each group after emit and broadcast, can be used to transform groups.
        Broadcasted values are available in the `broadcast` argument.

        Optional kwargs contain anything that was passed when calling the algorithm.
        """
        pass

    def __call__(self, rdd: RDD, **kwargs: Any) -> RDD:
        """
        Performs a single step of an algorithm, running all operations in sequence
        and ensuring data is partitioned correctly.

        Any additional keyword arguments passed to this function will be available
        in all life-cycle functions of the step:
        - `group`
        - `emit_by_group`
        - `broadcast`
        - `step`

        **DO NOT OVERRIDE WHEN DEFINING CUSTOM STEPS.**
        """
        if rdd.getNumPartitions() != self._n_partitions:
            rdd = rdd.repartition(self._n_partitions)

        step_cls: Type[Step] = self.__class__
        rdd = step_cls.group(
            rdd, **kwargs
        ).cache()  # cache because we use it twice (emit and step)

        def unwrap_emit(kv: Tuple[Any, Iterable[Any]]) -> Optional[Tuple[Any, Any]]:
            k, v = kv
            new_v = step_cls.emit_by_group(k, v, **kwargs)
            return new_v

        emitted = list(rdd.map(unwrap_emit).collect())
        to_broadcast = step_cls.broadcast(emitted, **kwargs)
        broadcast: Broadcast = self._sc.broadcast(to_broadcast)

        def unwrap_step(kv: Tuple[Any, Iterable[Any]]) -> Iterable[Any]:
            k, v = kv
            for new_v in step_cls.step(k, v, broadcast, **kwargs):
                yield new_v

        rdd = rdd.flatMap(unwrap_step)
        return rdd


class Algorithm(ABC):
    """
    `Algorithm` class provides a unified interface for executing steps (and other pyspark transformations),
    that form an entire algorithm.

    To implement a new algorithm, subclass `Algorithm` and implement `run` method.

    If you want to use steps in `run` method, list their classes in the `__steps__` dictionary inside your class,
    which for each key (`step_name`) and value (`StepClass`) will:
    - create an instance of `StepClass` with the same number of partitions as the parent algorithm instance
    - make this instance available as `step_name` instance variable in the `run` method for algorithm.

    For example, the `TeraSort` class uses its steps in the following way:
    ```python
    class TeraSort(Algorithm):
        # this dictionary maps instance variable name to a step class
        __steps__ = {
            "assign_buckets": SampleAndAssignBuckets,
            "sort": SortByKeyAndValue,
        }

        def run(self, rdd: RDD, n_dim: int) -> RDD:
            # because we defined `assign_buckets: SampleAndAssignBuckets` in `__steps__`,
            # the framework automatically created `self.assign_buckets` which is an instance
            # of SampleAndAssignBuckets class, which is a subclass of Step:
            rdd = self.assign_buckets(rdd, p=0.1, n_dim=n_dim)

            # simlarly, sort is an instance of a SortByKeyAndValue step:
            rdd = self.sort(rdd)
            return rdd

            # this was slightly simplified implementation of TeraSort than the real one,
            # feel free to check out the code in examples to see the differences
    ```

    To create an instance of `Algorithm`, you need to provide it with spark context and desired number of partitions.

    Calling instance of a `Algorithm` class will execute the `run` method in the desired environment
    (with inputs separated between partitions).

    For example, we can execute the `TeraSort` implementation above in the following way:
    ```python
    # create an instance of TeraSort:
    tera_sort = TeraSort(spark_context, n_partitions=2)

    # create some input data:
    input_rdd = spark_context.parallelize(range(10, 0, -1))

    # run the algorithm on the input data:
    output_rdd = tera_sort(input_rdd)
    ```
    """

    __steps__: Dict[str, Type[Step]] = dict()

    def __init__(self, sc: SparkContext, n_partitions: int):
        """
        Initializes instance of the algorithm.

        Every key in __steps__ dict will be mapped to an instance of corresponding
        value (which should be asubclass of Step)

        **DO NOT OVERRIDE WHEN DEFINING CUSTOM ALGORITHMS.**
        """

        self._n_partitions = n_partitions
        self._sc = sc

        for step_name, step_cls in self.__class__.__steps__.items():
            step_instance = step_cls(sc, n_partitions)
            setattr(self, step_name, step_instance)

        super().__init__()

    @property
    def n_partitions(self) -> int:
        return self._n_partitions

    @abstractmethod
    def run(self, **kwargs: Any) -> RDD:
        pass

    def __call__(self, **kwargs: Any) -> RDD:
        """
        Runs the algorithm, ensuring input data is partitioned correctly.

        All keyword arguments will be passed to run method.
        For now this class does not support non-keyword arguments.

        **DO NOT OVERRIDE WHEN DEFINING CUSTOM ALGORITHMS.**
        """
        for arg_name, arg in kwargs.items():
            if isinstance(arg, RDD) and arg.getNumPartitions() != self._n_partitions:
                kwargs[arg_name] = arg.repartition(self._n_partitions)

        rdd = self.run(**kwargs)
        return rdd
