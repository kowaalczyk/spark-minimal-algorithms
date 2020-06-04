FROM matthewfeickert/docker-python3-ubuntu:3.7.3

USER root

# install java, scala, openssh
RUN apt update
RUN apt -y dist-upgrade
RUN apt install -y openssh-server
RUN apt install -y openjdk-8-jdk
ENV JAVA_HOME "/usr/lib/jvm/java-8-openjdk-amd64/"
ENV JRE_HOME "/usr/lib/jvm/java-8-openjdk-amd64/jre/"
RUN apt install -y scala

# copy and install spark + hadoop
COPY ./spark-2.4.5-bin-hadoop2.7.tgz /usr/src/spark/spark-2.4.5-bin-hadoop2.7.tgz
RUN tar -xzf /usr/src/spark/spark-2.4.5-bin-hadoop2.7.tgz -C /usr/src/spark
RUN rm /usr/src/spark/spark-2.4.5-bin-hadoop2.7.tgz
ENV SPARK_HOME="/usr/src/spark/spark-2.4.5-bin-hadoop2.7"

# work in the project directory from now on
WORKDIR "/usr/src/spark-minimal-algorithms"

# install & configure poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# copy poetry settings and install dependencies
ADD pyproject.toml poetry.lock /usr/src/spark-minimal-algorithms/
RUN poetry install --no-interaction --no-ansi

# add project source code and tests
ADD .flake8 .gitignore mypy.ini README.md /usr/src/spark-minimal-algorithms/
ADD tests /usr/src/spark-minimal-algorithms/tests
ADD spark_minimal_algorithms /usr/src/spark-minimal-algorithms/spark_minimal_algorithms

# configure necessary environment variables for pyspark
ENV PYTHONPATH="${SPARK_HOME}/python:$PYTHONPATH"
ENV PYSPARK_PYTHON="python3.7"
ENV PYSPARK_DRIVER_PYTHON="python3.7"
ENV PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"
ENV PATH="${SPARK_HOME}:${SPARK_HOME}/bin:${PATH}:~/.local/bin:${JAVA_HOME}/bin:${JAVA_HOME}/jre/bin"
