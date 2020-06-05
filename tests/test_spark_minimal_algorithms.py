from spark_minimal_algorithms import __version__


def test_version():
    assert __version__ == "0.1.0"


def test_spark(spark_context):
    # spark_context is a pytest fixture provided by pytest-spark plugin
    assert spark_context.parallelize(range(3)).collect() == [0, 1, 2]
