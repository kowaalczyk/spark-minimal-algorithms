FROM kowaalczyk/pyspark-dev:latest

# work in the project directory from now on
WORKDIR "/usr/src/spark-minimal-algorithms"

# copy poetry settings and install dependencies
ADD pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi

# add project source code and tests
ADD .flake8 mypy.ini pytest.ini ./
ADD tests ./tests
ADD spark_minimal_algorithms ./spark_minimal_algorithms
