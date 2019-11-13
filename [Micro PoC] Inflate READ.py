# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Step 1: Load Source Data

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/04.avro"
file_type = "avro"

options = {}

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .options(**options) \
  .load(file_location)

# See source data from avro
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Step 2. Define and register UDFs to DEFLATE/INFLATE column

# COMMAND ----------

from pyspark.sql.functions import *
import zlib

@udf("string")
def tcs_inflate(body):
  try:
    inflated = zlib.decompress(body, wbits=-8).decode()
  except:
    inflated = "==ERROR=="
  return inflated

@udf("binary")
def tcs_deflate(body):
  return zlib.compress(body.encode())

spark.udf.register("tcs_inflate", tcs_inflate)
spark.udf.register("tcs_deflate", tcs_deflate)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Step 3. Inflate the Body

# COMMAND ----------

df_inflated = df.selectExpr("tcs_inflate(body) as body_inflated")

# COMMAND ----------

df_inflated.show(2)

# COMMAND ----------

# Check for records 
df_inflated.where("body_inflated='==ERROR=='").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse JSON From Fields

# COMMAND ----------

# Convert to plain RDD
rdd_json = df_inflated.select("body_inflated").rdd.map(lambda r: r.body_inflated )

# Parse RDD as multi-JSON
df_read = spark.read.json(rdd_json)

# COMMAND ----------

df_read = spark.read.json(rdd_json)

# COMMAND ----------

df_read.printSchema()

# COMMAND ----------

df1.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Ingest avro data from raw/source zone into prepard
# MAGIC * Store in parquet format in prepared zone:
# MAGIC   - parquet format
# MAGIC   - partition by event_time, where event_time is the time from the source directory where the original avro comes from, e.g. /2018/04/01/12/25 ... is converted to a column, called event_time.
# MAGIC * Create external table on the stored data.

# COMMAND ----------


