import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;

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
 --class LogisticRegressionAnalysis \
 --master yarn-cluster \
 stage3LG.jar \
 A2out1/
*
*/

public class LogisticRegressionAnalysis{

  public static void main(String[] args) {

    String outputFilePath = args[0];

    SparkConf conf = new SparkConf();
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<String> data = sc.textFile(
      "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-28x28.csv");

    JavaRDD<LabeledPoint> parsedData = data.map(line -> {
    String[] features = line.split(",");
    double[] v = new double[features.length];
    for (int i = 0; i < features.length - 1; i++) {
      v[i] = Double.parseDouble(features[i]);
    }
    return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
    });

    StructType schema = new StructType(new StructField[]{
      new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
    });

    Dataset<Row> df = spark
      .read()
      .schema(schema)
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-28x28.csv");
    // Converts Dataset to JavaRDD
    JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
      new Tuple2<>(model.predict(p.features()), p.label()));

    LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training.rdd());

    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
    double accuracy = metrics.accuracy();
    System.out.println("Accuracy = " + accuracy);

    model.save(sc, "target/tmp/javaLogisticRegressionWithLBFGSModel");
    LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc,
      outputFilePath);

    spark.stop();
  }
}
