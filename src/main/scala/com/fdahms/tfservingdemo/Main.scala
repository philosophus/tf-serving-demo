package com.fdahms.tfservingdemo

import io.grpc.ManagedChannelBuilder
import tensorflow.serving.PredictionServiceGrpc
import tensorflow.serving.Predict.PredictRequest
import tensorflow.serving.Model.ModelSpec
import java.util.concurrent.TimeUnit;
import org.tensorflow.framework.TensorProto
import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorShapeProto
import scala.collection.JavaConverters._

object Main {

  val channel = ManagedChannelBuilder.forAddress("localhost", 8500)
    .usePlaintext(true) // by default SSL/TLS is used
    .build()

  val stub = PredictionServiceGrpc.newBlockingStub(channel)

  def main(args: Array[String]):Unit = {

    print("Application of the magic model yields: ")
    println(applyModel(Seq(1.0, 6.0), Seq(18.0, 107.0)).mkString(", "))

    channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
  }


  def createTensorProto(values: Seq[Double]) = {
    val dim = TensorShapeProto.Dim.newBuilder()
      .setSize(values.size)

    val shape = TensorShapeProto.newBuilder()
      .addDim(dim)

    val builder = TensorProto.newBuilder()
      .setDtype(DataType.DT_FLOAT)
      .setTensorShape(shape)

    values.foreach{ value =>
      builder.addFloatVal(value.toFloat)
    }
    builder.build()
  }

  def createRequest(x: Seq[Double], y: Seq[Double]) = {
    val modelSpec = ModelSpec.newBuilder()
      .setName("default")
      .setSignatureName("magic_model")

    val requestBuilder = PredictRequest.newBuilder()
      .setModelSpec(modelSpec)
      .putInputs("egg", createTensorProto(x))
      .putInputs("bacon", createTensorProto(y))

    requestBuilder.build()
  }

  def applyModel(x: Seq[Double], y: Seq[Double]): Seq[Double] = {
    val response = stub.predict(createRequest(x, y))
    response.getOutputsOrThrow("outputs")
      .getFloatValList()
      .asScala
      .map(_.toDouble)
  }
}
