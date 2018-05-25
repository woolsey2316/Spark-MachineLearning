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
*/

public class MultilayerPerceptionClassifierAnalysis {
  public static void main(String[] args) {

    String outputFilePath = args[0];

    // Load training data
    SparkSession spark = SparkSession
    .builder()
    .appName("Logistic Regression")
    .getOrCreate();

    StructType schema = new StructType(new StructField[]{
      new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
    });

    Dataset<Row> df = spark
      .read()
      .schema(schema)
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv");

    PCAModel pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(df);

      int[] layers = new int[] {3, 4, 4, 2};

      // Trainer phase
      MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(30)
        .setSeed(1234L)
        .setMaxIter(100);

      // train the model
      MultilayerPerceptronClassificationModel model = trainer.fit(pca);

      // compute accuracy on the test set
      Dataset<Row> result = model.transform(df);
      Dataset<Row> predictionAndLabels = result.select("prediction", "label");
      MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy");

      // Test phase
      Dataset<Row> test = spark
        .read()
        .schema(schema)
        .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv");

      Dataset<Row> results = model.transform(test);

      for (Row r: rows.collectAsList()) {
        System.out.println("(" + r.get(0) + ", " + r.get(1) + ") -> prob=" + r.get(2)
          + ", prediction=" + r.get(3));
      }

      System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
    spark.stop();
  }
}
