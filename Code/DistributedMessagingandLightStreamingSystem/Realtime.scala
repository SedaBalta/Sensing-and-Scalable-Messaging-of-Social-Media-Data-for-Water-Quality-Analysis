package classification

import org.apache.spark.sql.{DataFrame, SparkSession, functions => F}
import org.apache.spark.sql.types._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, NaiveBayesModel}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer, VectorAssembler}
import org.apache.spark.sql.streaming._
import org.apache.spark.{SparkConf, SparkContext}

class lrMLSinkProvider extends MLSinkProvider{
  override def process(df: DataFrame): Unit = {
    LRModel.transform(df)
  }
}

object Realtime {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    // the directory where we store the testing csv file
    val fileDir= "C:\\Users\\ASUS\\Desktop\\IntelIJ\\test"

    val spark: SparkSession  = SparkSession.builder()
      .appName("Realtime")
      .master("local[4]")
      .config("spark.driver.memory", "2g")
      .config("spark.executor.memory", "4g")
      .getOrCreate()



    val schema = StructType(
      Array(StructField("id", DoubleType),
        StructField("Unnamed: 0", DoubleType),
        StructField("Unnamed: 0.1", DoubleType),
        StructField("text", StringType),
        StructField("Subjectivity", DoubleType),
        StructField("Polarity", DoubleType),
        StructField("Analysis", StringType),
        StructField("label", DoubleType)
      ))

    val df = spark
      .readStream
      .option("header", "true")
      .schema(schema)
      .csv(fileDir)


    val dropdf =df.drop("Unnamed: 0").drop("Unnamed: 0.1").drop("Subjectivity").drop("Polarity").drop("Analysis").drop("id")
   // dropdf.show()

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val tokenizedDF = tokenizer.transform(dropdf)
    //tokenizedDF.show()

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val hashingDF = hashingTF.transform(tokenizedDF)

    val pipeModel = PipelineModel.load("C:\\Users\\ASUS\\Desktop\\modelbatch2")

    //structured streaming
    val streamModel = pipeModel.stages.last.asInstanceOf[LogisticRegressionModel]
    val transformedDF = streamModel.transform(hashingDF)
    val checkPointDir = "C:\\Users\\ASUS\\Desktop\\Checkpoint"


    //This function can distrubition by topic name via Apache Kafka
    def myCustomFunc(df:DataFrame, batchID:Long): Unit ={
      // Prediction column= "0" topic name color
      val colorDF = df.filter((F.col("prediction").equalTo(0.0)))
      val colorDF2 = colorDF.withColumn("value",
        F.concat(
          F.col("text"),F.lit(","),
          F.col("label"),F.lit(","),
          F.col("prediction")
        ))

      colorDF2.select("value").write
        .format("kafka")
        .option("kafka.bootstrap.servers","localhost:9092")
        .option("topic","color")
        .save()

      val illnessDF = df.filter((F.col("prediction").equalTo(1.0)))
      val illnessDF2 = illnessDF.withColumn("value",
        F.concat(
          F.col("text"),F.lit(","),
          F.col("label"),F.lit(","),
          F.col("prediction")
        ))

      illnessDF2.select("value").write
        .format("kafka")
        .option("kafka.bootstrap.servers","localhost:9092")
        .option("topic","illness")
        .save()

      val odortasteDF = df.filter((F.col("prediction").equalTo(2.0)))
      val odortasteDF2 = odortasteDF.withColumn("value",
        F.concat(
          F.col("text"),F.lit(","),
          F.col("label"),F.lit(","),
          F.col("prediction")
        ))

      odortasteDF2.select("value").write
        .format("kafka")
        .option("kafka.bootstrap.servers","localhost:9092")
        .option("topic","odortaste")
        .save()

      val usDF = df.filter((F.col("prediction").equalTo(3.0)))
      val usDF2 = usDF.withColumn("value",
        F.concat(
          F.col("text"),F.lit(","),
          F.col("label"),F.lit(","),
          F.col("prediction")
        ))

      usDF2.select("value").write
        .format("kafka")
        .option("kafka.bootstrap.servers","localhost:9092")
        .option("topic","unusualstate")
        .save()
    }

    val query = transformedDF.writeStream
      .foreachBatch(myCustomFunc _)
      .option("checkpointLocation", checkPointDir)
      .start()

    query.awaitTermination()

  }
}
