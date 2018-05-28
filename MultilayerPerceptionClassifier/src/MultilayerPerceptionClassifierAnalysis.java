import java.util.Arrays;
import java.util.List;
import java.io.IOException;

import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;

import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.PCA;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.Metadata;

/**
*submit to a yarn cluster
*
spark-submit  \
 --class MultilayerPerceptionClassifierAnalysis \
 --master local[4] \
 --driver-memory 4g \
 --num-executors 4 \
 Stage3MPC.jar \
 A2out2/
*
Multilayer perceptron classifier (MLPC) is a kind of feedforward
artificial neural network. MLPC has many layers of nodes. Each layer
is fully connected to the next layer in the network. Nodes in the input layer
represent the input data. The MLP consists of three or more layers (an input and
an output layer with one or more hidden layers) of nonlinearly-activating nodes
making it a deep neural network. Since MLPs are fully connected, each node in one
layer connects with a certain weight to every node in the following layer.
*/

public class MultilayerPerceptionClassifierAnalysis {
  public static void main(String[] args) {

    String outputFilePath = args[0];

    // Load training data
    SparkSession spark = SparkSession
    .builder()
    .appName("Multilayer Perception Classifier")
    .getOrCreate();

    StructType schema = new StructType(new StructField[]{
      new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
    });

    Dataset<Row> trainingData = spark
      .read()
      .option("inferschema","true")
      .csv("hdfs://soit-hdp-pro-8.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv");

    Dataset<Row> testingData = spark
      .read()
      .option("inferschema","true")
      //.csv("hdfs://soit-hdp-pro-11.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv");
      .csv("Test-label-28x28.csv");

    int[] layers = new int[] {3, 4, 4, 2};

    // Trainer phase
    MultilayerPerceptronClassifier multilayerPerceptronClassifier = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      // We are assigned to examine the various block sizes
      .setSeed(1234L)
      .setMaxIter(100);

    ParamMap[] paramGrid = new ParamGridBuilder()
      .addGrid(multilayerPerceptronClassifier.blockSize(), new int[] {5, 15, 30})
      .build();

    TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
      .setEstimator(multilayerPerceptronClassifier)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid);

    TrainValidationSplitModel model = trainValidationSplit
      .fit(trainingData);

    // compute accuracy on the test set
    Dataset<Row> evaluatePrediction = model.transform(testingData).select("prediction", "label");
      MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy");

    try {
      evaluatePrediction.save(outputFilePath + "logreg.csv");
    }
    catch(IOException e) {
      e.printStackTrace();
    }

    spark.stop();
  }
}
