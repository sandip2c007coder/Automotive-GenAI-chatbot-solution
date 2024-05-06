# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configure common variables
# MAGIC %run ./Reusable_Variable_Config_CodeGen

# COMMAND ----------


vs_index_fullname=VECTOR_INDEX_FULL_NAME
print(vs_index_fullname)

print(VECTOR_SEARCH_ENDPOINT_NAME)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

