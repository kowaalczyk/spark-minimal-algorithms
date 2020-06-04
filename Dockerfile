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

# install & configure poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

ENV SPARK_HOME="/usr/src/spark/spark-2.4.5-bin-hadoop2.7"

# copy the project to a working directory
ADD .flake8 .gitignore mypy.ini pyproject.toml README.md /usr/src/spark-minimal-algorithms/
ADD tests /usr/src/spark-minimal-algorithms/tests
ADD spark_minimal_algorithms /usr/src/spark-minimal-algorithms/spark_minimal_algorithms
WORKDIR "/usr/src/spark-minimal-algorithms"

RUN poetry install

# configure necessary environment variables for pyspark
ENV PYTHONPATH="${SPARK_HOME}/python:$PYTHONPATH"
ENV PYSPARK_PYTHON="python3.7"
ENV PYSPARK_DRIVER_PYTHON="python3.7"
ENV PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"
ENV PATH="${SPARK_HOME}:${SPARK_HOME}/bin:${PATH}:~/.local/bin:${JAVA_HOME}/bin:${JAVA_HOME}/jre/bin"
