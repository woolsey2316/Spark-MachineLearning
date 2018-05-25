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
    SparkSession spark = SparkSession
    .builder()
    .appName("Logistic Regression")
    .getOrCreate();

    String outputFilePath = args[0];

    JavaRDD<LabeledPoint> parsedData = data.map(row -> {
    String[] features = row.split(",");
    int[] v = new int[features.length];
    for (int i = 0; i < features.length - 1; i++) {
      v[i] = Double.parseDouble(features[i]);
    }
    return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
    });

    PCAModel pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(df);

    Dataset<Row> result = pca.transform(df).select("pcaFeatures");
    result.show(false);

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
