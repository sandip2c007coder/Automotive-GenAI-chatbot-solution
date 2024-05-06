# Databricks notebook source
# MAGIC %md 
# MAGIC *Note: RAG performs document searches using Databricks Vector Search. In this notebook, we assume that the search index is ready for use.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install the required libraries
# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 databricks-sdk==0.12.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configure common variables
# MAGIC %run ./Reusable_Variable_Config_CarModel

# COMMAND ----------

host =DECLARED_HOST
decleared_databricks_token=DECLARED_DATABRICKS_TOKEN
print(host)

index_name=VECTOR_INDEX_FULL_NAME
print(index_name)

print(VECTOR_SEARCH_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Langchain retriever
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question (our Managed Vector Search Index will compute the embeddings for us)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup authentication for our model
import os
# url used to send the request to your model from the serverless endpoint
#host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = decleared_databricks_token   #dbutils.secrets.get("dbdemos", "rag_sp_token")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
#print(f"Test embeddings: {embedding_model.embed_query('Find the task list from the technical document ?')[:20]}...")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=decleared_databricks_token,disable_notice=True)
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()


# test our retriever --- this is also not required
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("How many positions are there in CEILING LIGHT of Nissan Altima?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Building Databricks Chat Model 
# MAGIC Our chatbot will be using dbrx-instruct model to provide answer. 
# MAGIC
# MAGIC

# COMMAND ----------

# Test Databricks Foundation LLM model -- This is not returning results from Vector DB
from langchain.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct")#, max_tokens = 500)
#chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)
print(f"Test chat model: {chat_model.predict('How many positions are there in CEILING LIGHT of Nissan Altima?')}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Assembling the complete RAG Chain
# MAGIC Let's now merge the retriever and the model in a single Langchain chain.
# MAGIC
# MAGIC We will use a custom langchain template for our assistant to give proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

TEMPLATE = CUSTOM_LANGCHAIN_TEMPLATE

prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# DBTITLE 1,Let's try our chatbot in the notebook directly:
# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"query": " How many positions are there in CEILING LIGHT of Nissan Altima?"}
answer = chain.run(question)
print(answer)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving our model to model registry
# MAGIC

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

model_name = MODEL_NAME
answer="This is the answer to the question"

with mlflow.start_run(run_name=MODEL_RUN_NAME) as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        #input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Deploying our Chat Model as a Serverless Model Endpoint 
# MAGIC
# MAGIC Not able to do this as we are using Trial workspace