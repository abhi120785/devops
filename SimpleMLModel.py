# Databricks notebook source
dbutils.library.installPyPI("mlflow", extras="extras")
dbutils.library.restartPython()

# COMMAND ----------

import mlflow

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Simple ML Model

# COMMAND ----------

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import mlflow.sklearn

# COMMAND ----------

rng = np.random.RandomState(1)
X = np.sort(5*rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

with mlflow.start_run(run_name="Compare Decision Tree Regression Max Depth"):
  regr_1 = DecisionTreeRegressor(max_depth=2)
  regr_2 = DecisionTreeRegressor(max_depth=5)
  
  mlflow.log_param("first_model", str(regr_1))
  mlflow.log_param("second_model", str(regr_2))

  regr_1.fit(X, y)
  regr_2.fit(X, y)
  
  mlflow.sklearn.log_model(regr_1, "first_fitted_model")
  mlflow.sklearn.log_model(regr_2, "second_fitted_model")
  
  # Predict
  X_test = np.arange(0.0, 5.0, 0.01)[:np.newaxis]
  y_1 = regr_1.predict(X_test.reshape(-1, 1))
  y_2 = regr_2.predict(X_test.reshape(-1, 1))
  
  plt.figure()
  plt.scatter(X, y, s=20, edgecolor="black",
             c="darkorange", label="data")
  plt.plot(X_test, y_1, color="cornflowerblue",
          label="maxdept=2", linewidth=2)
  plt.plot(X_test, y_2, color="yellowgreen", 
          label="maxdepth=1", linewidth=2)
  plt.xlabel("data")
  plt.ylabel("target")
  plt.title("Decision Tree Regressoin")
  plt.legend()
  
  plt.savefig("./example_fig.png")
  mlflow.log_artifact("./example_fig.png")
  plt.show()
  display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transforming DataFrame in Spark Using ML Model and UDF
# MAGIC 
# MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/156434371205756/4245485951139576/4056451362173976/latest.html

# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql import Row

@udf("float")
def udf_predict(x):
    prediction = regr_2.predict(np.array(float(x)).reshape(-1,1))
    return float(prediction[0])

# COMMAND ----------

test_df = spark.createDataFrame(map(lambda x: Row(x=float(x)), X_test))

# COMMAND ----------

test_df.show(5)

# COMMAND ----------

predicted_df = test_df.withColumn('y', udf_predict(col("x")))
predicted_df.show(5)

# COMMAND ----------


