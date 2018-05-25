import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.SharedSparkSession;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class MultiLayerPerceptionClassifierAnalysis {
  public static void main(String[] args) {
    // Load training data
    String inputFilePath = args[0], outputFilePath = args[1];

    String outputFilePath = args[0];

    SparkConf conf = new SparkConf();
    JavaSparkContext sc = new JavaSparkContext(conf);
    JavaRDD<String> data = sc.textFile(
      "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-28x28.csv");

    JavaRDD<LabeledPoint> parsedData = data.map(row -> {
    String[] features = row.split(",");
    int[] v = new int[features.length];
    for (int i = 0; i < features.length - 1; i++) {
      v[i] = Double.parseDouble(features[i]);
    }
    return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
    });

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    int[] layers = new int[] {4, 5, 4, 3};

    // create the trainer and set its parameters
    MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(30)
      .setSeed(1234L)
      .setMaxIter(100);

    // train the model
    MultilayerPerceptronClassificationModel model = trainer.fit(train);

    // compute accuracy on the test set
    Dataset<Row> result = model.transform(test);
    Dataset<Row> predictionAndLabels = result.select("prediction", "label");
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy");

    System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
    spark.stop();
  }
}
