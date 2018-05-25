import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.distributed.RowMatrix;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.PCA;

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
 Stage3LG.jar \
 A2out1/
*
*/

public class LogisticRegressionAnalysis{

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

    Dataset<Row> result = pca
      .transform(df)
      .select("pcaFeatures");

    result.show(5);

    LogisticRegression lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01);

    LogisticRegressionModel model = lr.fit(pca);

    Vector weights = model.weights();

    Dataset<Row> test = spark
      .read()
      .schema(schema)
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv");

    Dataset<Row> result = model.transform(test).show();

    spark.stop();
  }
}
