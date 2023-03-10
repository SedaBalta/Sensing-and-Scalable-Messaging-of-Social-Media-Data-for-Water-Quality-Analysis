package classification
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

object LRModel {
  private val modelPath = "C:\\Users\\ASUS\\Desktop\\modelbatch2"

  // load the pre trained model
  private val model = PipelineModel.read.load(modelPath)

  def transform(df: DataFrame) = {
    // replace nan values with 0
    val df_clean = df.na.fill(0)

    // run the model on new data
    val result = model.transform(df_clean)

    // display the results
    result.show()
  }
}
