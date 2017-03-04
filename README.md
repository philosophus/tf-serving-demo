# TensorFlow Serving Demo

This is a small demo project to show how a JVM project can use TensorFlow models which are served by TensorFlow Serving.

## Requirements

* [TensorFlow Serving](https://tensorflow.github.io/serving/setup)

## How to run

Start Tensorflow Serving using the model

    [path to tensorflow_serving]/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --model_base_path="model/three_x_plus_y"

Run the example client

    ./gradlew run
