import java.util.Arrays;
import java.util.List;
import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;

import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.PCA;

import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.Metadata;

import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;

/**
*submit to a yarn cluster
*
spark-submit  \
 --class LogisticRegressionAnalysis \
 --master local[2] \
 --driver-memory 4g \
 --num-executors 4 \
 --executor-cores 5 \
 Stage3LG.jar \
 A2out1/
*
single line
*
spark-submit --class LogisticRegressionAnalysis --master yarn-cluster Stage3LG.jar A2out1/
*/

public class LogisticRegressionAnalysis{

  public static void main(String[] args) {

    // Create spark session
    SparkSession spark = SparkSession
      .builder()
      .appName("Logistic Regression")
      .getOrCreate();

    String outputFilePath = args[0];
    /*
    Machine learning algorithms expect input data to be in a single vector that
    represents all the features of the data point. The schema below takes 748
    columns each of integer type and produces a single column of vector type.
    This column is called "features".
    */
    StructType schema = new StructType(new StructField[]{
      new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("pixel", DataTypes.StringType, false, Metadata.empty()),
    });

// Fit the pipeline to training documents.
    /*
    Spark will know to use SparseVector format for representing the image pixel values.
    In Java, a DataFrame is represented by a Dataset of Rows
    */
    Dataset<Row> trainingData = spark
      .read()
      .option("inferschema","true")
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv");

      trainingData.show();

    Dataset<Row> testingData = spark
      .read()
      .option("inferschema","true")
      .csv("hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv");

    /*
      Principal component analysis transforms a set of observations into a set of orthogonal components. When removing
      Principal componets, start with those with the lowest variance in order to minimise inforation loss. In regression analysis
      One approach, especially when there are strong correlations between different possible explanatory variable  is to reduce them
      to a few principal components and then run the regression against them. project the 748 feature vectors into 3-dimensional principal components
     */
    PCA principalComponents = new PCA()
      .setInputCol("features")
      //only want the transformed vector, number of columns = 1
      .setOutputCol("featuresPCA");

    ParamMap[] paramGrid = new ParamGridBuilder()
      .addGrid(principalComponents.k(), new int[] {5, 20, 50, 748})
      .build();

    LogisticRegression logreg = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01);

    TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
      .setEstimator(logreg)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid);

    TrainValidationSplitModel model = trainValidationSplit
      .fit(trainingData);

    model.transform(testingData);

    try {
      model.save(outputFilePath + "LogisticRegressionModel.csv");
    }
    catch(IOException e) {
      e.printStackTrace();
    }
    spark.stop();
  }
}
