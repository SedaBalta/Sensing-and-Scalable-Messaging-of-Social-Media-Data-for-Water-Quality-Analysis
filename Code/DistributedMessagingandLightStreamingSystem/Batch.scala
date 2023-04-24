package classification
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}


object Batch {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark  = SparkSession.builder()
      .appName("Batch")
      .master("local[4]")
      .config("spark.driver.memory", "2g")
      .config("spark.executor.memory", "4g")
      .getOrCreate()

    val filePath= "C:\\Users\\ASUS\\Desktop\\datasets\\train-allclasses.csv"
    val modelPath = "C:\\Users\\ASUS\\Desktop\\modelbatch3"

    val schema = StructType(
      Array(StructField("id", DoubleType),
        StructField("Unnamed: 0", DoubleType),
        StructField("Unnamed: 0.1", DoubleType),
        StructField("text", StringType),
        StructField("Subjectivity", DoubleType),
        StructField("Polarity", DoubleType),
        StructField("Analysis", StringType),
	StructField("Coordinates", StringType),
        StructField("label", DoubleType)
      ))

    val df_raw = spark
      .read
      .option("header", "true")
      .schema(schema)
      .csv(filePath)

    val df = df_raw.na.fill(0)
    df.show()

    val dropdf =df.drop("Unnamed: 0").drop("Unnamed: 0.1").drop("Subjectivity").drop("Polarity").drop("Analysis").drop("id")
    dropdf.show()

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val tokenizedDF = tokenizer.transform(dropdf)
    tokenizedDF.show()

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val hashingDF = hashingTF.transform(tokenizedDF)

    val logregObj = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, logregObj))

    val cvModel = pipeline.fit(dropdf)

    //save the model
    cvModel.write.overwrite().save(modelPath)
    println("Saved")
  }
}
