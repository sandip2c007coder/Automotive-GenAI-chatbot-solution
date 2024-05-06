# Databricks notebook source
# DBTITLE 1,Input/configuration needed
RAG_SUBJECT='carmodel'
DECLARED_DATABRICKS_TOKEN="dapiXXXXXXXXXXXXXXXX"

CUSTOM_LANGCHAIN_TEMPLATE = """You are an assistant for describing the details of CarModel. You are answering queries on Car Manuals which is used to train you. If the question is not related to these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Don't try to send any incomplete answer.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""

# COMMAND ----------

# DBTITLE 1,Derrived variables


EXTERNAL_VOLUME_NAME= 'ExternalVolume_'+RAG_SUBJECT
CATALOG_SCHEMA='genai.vector_db.'
EXTERNAL_VOLUME_FULL_NAME=CATALOG_SCHEMA+EXTERNAL_VOLUME_NAME
EXTERNAL_VOLUME_PATH='s3://databrickscoe/'+RAG_SUBJECT+'/UploadedFile'


PDF_PATH = '/Volumes/genai/vector_db/'+EXTERNAL_VOLUME_NAME
TABLE_NAME_FOR_FILE_LOAD=RAG_SUBJECT
FULL_TABLE_NAME_FOR_FILE_LOAD=CATALOG_SCHEMA+TABLE_NAME_FOR_FILE_LOAD


VECTOR_SEARCH_ENDPOINT_NAME= RAG_SUBJECT+'_vector_endpoint'
VECTOR_INDEX_FULL_NAME = CATALOG_SCHEMA+RAG_SUBJECT+'_vs_index'

DECLARED_HOST ="https://" + spark.conf.get("spark.databricks.workspaceUrl")
MODEL_NAME=RAG_SUBJECT+'_rag_model'
MODEL_RUN_NAME=RAG_SUBJECT+'_rag_model_run'

# COMMAND ----------



# COMMAND ----------

print(EXTERNAL_VOLUME_FULL_NAME)
print(EXTERNAL_VOLUME_PATH)
print(PDF_PATH)
print(FULL_TABLE_NAME_FOR_FILE_LOAD)
print(VECTOR_SEARCH_ENDPOINT_NAME)
print(VECTOR_INDEX_FULL_NAME)
print(DECLARED_HOST)
print(MODEL_NAME)
print(MODEL_RUN_NAME)