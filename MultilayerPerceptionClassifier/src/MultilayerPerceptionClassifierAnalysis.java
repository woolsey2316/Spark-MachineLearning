import java.util.Arrays;
import java.util.List;

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
 --master yarn-cluster \
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

    Dataset<Row> dataFrame = spark
      .read()
      .schema(schema)
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv");
    //PCA reduction may not be needed for this algorithm
    PCAModel pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(dataFrame);

	int[] layers = new int[] {3, 4, 4, 2};

	// Trainer phase
	MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
	  .setLayers(layers)
	  // We are assigned to examine the various block sizes
	  .setBlockSize(30)
	  .setSeed(1234L)
	  .setMaxIter(100);

	// train the model
	MultilayerPerceptronClassificationModel model = trainer.fit(pca);

	// compute accuracy on the test set
	Dataset<Row> evaluatePrediction = model.transform(df).select("prediction", "label");
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
	  .setMetricName("accuracy");

	dataFrame.write.repartition(1).format("com.MultilayerPerceptron.spark.csv").option("header", "true").save(outputFilePath + ".csv");
    spark.stop();
  }
}
