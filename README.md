# Minimal Algorithms for Apache Spark

An framework for implementing [Minimal Mapreduce Algorithms](https://www.cse.cuhk.edu.hk/~taoyf/paper/sigmod13-mr.pdf)
for Apache Spark using Python.


## Motivation

The paper [Minimal Mapreduce Algorithms](https://www.cse.cuhk.edu.hk/~taoyf/paper/sigmod13-mr.pdf) introduces
a good way to reason about distributed algorithms. Algorithm is minimal if each machine executing the algorithm:
- uses *O(n/t)* memory
- sends/recieves *O(n/t)* pieces of information
- performs *O(Tseq/t)* amount of computation
where *t* is the number of machines in the cluster, *n* is the size of input data and *Tseq* is
the optimal computational complexity of sequential algorithm.

Essentially, the algorithm being minimal means we can achieve *t*-times speedup if we
use *t* machines instead of 1. This project was started as a class assignment, but can
serve as an efficient framework for implementing new minimal algorithms.

The framework and all the examples are implemented in Python, and I think they can
be a good starting point for someone wanting to learn Apache Spark. That said, if you are here
to learn, I recommend reading [QuickStart](https://spark.apache.org/docs/latest/quick-start.html)
and [RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html) first.


## Quickstart

Install library:
```shell
pip install git+https://github.com/kowaalczyk/spark-minimal-algorithms.git
```

Algorithm documentation contains detailed description of its steps.
It can be accessed using standard python help built-in function, for example:
```python
from spark_minimal_algorithms.examples import TeraSort

help(TeraSort)
```

### Steps and algorithms

All minimal algorithms are just a few rounds of map, reduce and shuffle operations.
To provide a unified API for all algorithms, this framework uses `Step` and `Algorithm` classes.

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

`Step` and `Algorithm` classes also have some more advanced features, which are documented
in their respective docstrings.

The [`spark_minimal_algorithms.examples.Countifs` algorithm](spark_minimal_algorithms/examples/countifs.py)
is a good advanced example that uses nearly all features of the framework.


### Running an algorithm


## Contributing

While this project was originally created for an university course, I think it actually may be
a nice basis for learning about distributed systems and algorithms.

I will certainly appreciate any contributions to this project :)


### Project status

Examples - from original paper ["Minimal MapReduce Algorithms"](https://www.cse.cuhk.edu.hk/~taoyf/paper/sigmod13-mr.pdf):
- [x] TeraSort
- [ ] PrefixSum (this is pretty much implemented in COUNTIFS, but may be a nice simple example to start with)
- [ ] GroupBy
- [ ] SemiJoin
- [ ] SlidingAggregation

Examples - from the paper
[Towards minimal algorithms for big data analytics with spreadsheets](https://dl.acm.org/doi/10.1145/3070607.3075961):
- [ ] COUNTIF (Interval Multiquery Processor algorithm, which solves simplified 1D case of COUNTIFS)
- [x] COUNTIFS (Multidimensional Interval Multiquery Processor, solves problem for any number of dimensions)

Documentation:
- [x] framework docstrings
- [x] examples documentation
- [x] quickstart
- [ ] github workflow for automatically generating docs and publishing them as a github page
- [ ] tutorial on setting up a spark cluster + how to run examples and check performance

Developer experience:
- [ ] custom `__repr__` for step and algorithm, showing what exactly (step-by-step) happens in these classes
- [ ] add metaclass that will stop users from overriding `__init__` and `__call__` for `Algorithm` and `Step`


### Development setup

All necessary testing and linting can be run in docker, so you don't need to install
anythong locally.

After making changes to the code, make sure to re-build the docker image:
```shell
docker-compose build
```

This step can be long, but only for the 1st time.

After that, most common tasks have dedicated docker-compose services that can be run:
```shell
# run most tests very quickly:
docker-compose up test

# run all tests (including the long ones):
docker-compose up test-long

# run linter:
docker-compose up lint
```

The last 2 commands are also automatically performed by github workflow for every pull request to the master branch, so there is no need to run long tests locally.

Docker containers can be also used to run custom commands, for example:
```shell
# open bash and run anything you want inside the container:
docker-compose run test bash

# run tests from a single file, first failed test will open a debugger:
docker-compose run test poetry run pytest -x --pdb tests/examples/test_tera_sort.py
```

You may wish to setup poetry project locally (even without Apache Spark installation)
for your editor/ide to use of flake8 linter, black formatter and other tools with
the same settings as in docker (which are the same in CI).

To do this, you need Python 3.7 and [poetry](https://python-poetry.org/) installed.
Simply run `poetry install` to create a virtual environment and install all dependencies.
