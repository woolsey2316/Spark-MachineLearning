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
single line
*
spark-submit --class LogisticRegressionAnalysis --master yarn-cluster Stage3LG.jar A2out1/
*/

public class LogisticRegressionAnalysis{

  public static void main(String[] args) {

    String outputFilePath = args[0];
    // Create spark session
    SparkSession spark = SparkSession
    .builder()
    .appName("Logistic Regression")
    .getOrCreate();
    /**
    Machine learning algorithms expect input data to be in a single vector that
    represents all the features of the data point. The schema below takes 748 
    columns each of integer type and produces a single column of vector type. 
    This column is called "features".
    **/
    StructType schema = new StructType(new StructField[]{
      new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
    });
    /*
    Spark will know to use SparseVector format for representing the image pixel values.
    In Java, a DataFrame is represented by a Dataset of Rows
    */ 
    Dataset<Row> dataFrame = spark
      .read()
      .schema(schema)
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv");

    /*
      Principal component analysis transforms a set of observations into a set of orthogonal components. When removing 
      Principal componets, start with those with the lowest variance in order to minimise inforation loss. In regression analysis 
      One approach, especially when there are strong correlations between different possible explanatory variable  is to reduce them 
      to a few principal components and then run the regression against them. project the 748 feature vectors into 3-dimensional principal components
     */
    PCAModel pca = new PCA()
      .setInputCol("features")
      //only want the transformed vector, number of columns = 1
      .setOutputCol("featuresPCA")
      //number of principal components
      .setK(3)
      //trains pca with given data set
      .fit(dataFrame);

    // result has k principal components in its vector
    Dataset<Row> result = principleComponents
      //the trained model performs the actual transformation
      .transform(dataFrame)
      .select("featuresPCA");

    LogisticRegressionModel model = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01);
      .fit(principleComponents);

    Dataset<Row> test = spark
      .read()
      .schema(schema)
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv");

    Dataset<Row> result = model
      .transform(test)
      .show();

    dataFrame.write.repartition(1).format("com.logisticregression.spark.csv").option("header", "true").save(outputFilePath + ".csv");

    spark.stop();
  }
}
