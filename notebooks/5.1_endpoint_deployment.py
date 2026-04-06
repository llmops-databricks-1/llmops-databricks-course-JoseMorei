# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 5.1: Endpoint Deployment
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating Model Serving endpoints
# MAGIC - Endpoint configuration and scaling
# MAGIC - Deploying models from Unity Catalog
# MAGIC - Testing deployed endpoints
# MAGIC - Using the serving utilities from arxiv_curator

# COMMAND ----------
import time
from databricks.sdk import WorkspaceClient
import mlflow
from pyspark.sql import SparkSession
from loguru import logger

from arxiv_curator.config import load_config, get_env

# COMMAND ----------

# Setup
import os
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

w = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

logger.info(f"Environment: {env}")
logger.info(f"Catalog: {cfg.catalog}")
logger.info(f"Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Model Serving Endpoints Overview
# MAGIC
# MAGIC **Databricks Model Serving** provides:
# MAGIC - Scalable, low-latency inference
# MAGIC - Auto-scaling based on traffic
# MAGIC - A/B testing and traffic splitting
# MAGIC - Inference tables for monitoring
# MAGIC - Built-in authentication and rate limiting
# MAGIC
# MAGIC ### Endpoint Types:
# MAGIC
# MAGIC - **Provisioned Throughput**: Reserved capacity, predictable performance
# MAGIC - **Serverless**: Auto-scaling, pay-per-request
# MAGIC
# MAGIC ### For GenAI Agents:
# MAGIC
# MAGIC - Use **Serverless** for development and variable traffic
# MAGIC - Use **Provisioned** for production with consistent load
# MAGIC - Enable **Inference Tables** for observability

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy Agent Using agents.deploy()
# MAGIC
# MAGIC The `agents.deploy()` API simplifies deployment by:
# MAGIC - Automatically configuring the endpoint
# MAGIC - Setting up inference tables
# MAGIC - Managing environment variables
# MAGIC - Handling model versioning

# COMMAND ----------

from databricks import agents
from mlflow import MlflowClient

# Get model details
model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "latest-model").version
endpoint_name = f"arxiv-agent-{env}"

# Get experiment ID for tracing
experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info(f"Deploying agent:")
logger.info(f"  Model: {model_name}")
logger.info(f"  Version: {model_version}")
logger.info(f"  Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the Agent

# COMMAND ----------

# Deploy agent to serving endpoint
agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    scale_to_zero=True,
    workload_size="Small",
    environment_vars={
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
    },
)

logger.info(f"✓ Deployment complete!")
logger.info(f"  Endpoint: {endpoint_name}")
logger.info(f"  Model: {model_name}@{model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Deployed Endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for Endpoint to be Ready
# MAGIC
# MAGIC The deployment may take 5-10 minutes. You can check the endpoint status in the UI or wait here.

# COMMAND ----------

# Wait for endpoint to be ready (optional - can also check in UI)
# w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name)
# logger.info(f"✓ Endpoint is ready!")

# MAGIC %md
# MAGIC ### Send Test Request

# COMMAND ----------

# Test request using the agent's expected format
test_request = {
    "input": [
        {
            "role": "user",
            "content": "What are recent papers about vision transformers?"
        }
    ]
}

logger.info("Sending request to endpoint...")

# Use the workspace client's API client for authenticated requests
api_client = w.api_client
endpoint_url = f"/api/2.0/serving-endpoints/{endpoint_name}/invocations"

response = api_client.do(
    method="POST",
    path=endpoint_url,
    body=test_request
)

logger.info("\n✓ Response received:")
if "output" in response:
    logger.info(f"\n  Content:")
    for item in response["output"]:
        if isinstance(item, dict) and "content" in item:
            logger.info(f"  {item['content']}")
        else:
            logger.info(f"  {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Streaming Response

# COMMAND ----------

logger.info("Testing streaming response...")

# Streaming request
stream_request = {
    "input": [
        {
            "role": "user",
            "content": "Summarize recent work on large language models"
        }
    ],
    "stream": True
}

# For streaming, we need to use requests with proper authentication
import os
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    # On Databricks, use notebook context
    from databricks.sdk.runtime import dbutils
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
else:
    # Local execution
    token = w.config.token

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"{w.config.host}/api/2.0/serving-endpoints/{endpoint_name}/invocations",
    headers=headers,
    json=stream_request,
    stream=True
)

logger.info("\n✓ Streaming response:")
for line in response.iter_lines():
    if line:
        try:
            data = json.loads(line.decode('utf-8').replace('data: ', ''))
            if isinstance(data, dict) and 'item' in data:
                item = data['item']
                if isinstance(item, dict) and 'content' in item:
                    print(item['content'], end="", flush=True)
        except:
            pass

logger.info("\n\n✓ Streaming complete")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Monitor Endpoint Performance

# COMMAND ----------

# Get endpoint metrics
endpoint_details = w.serving_endpoints.get(endpoint_name)

logger.info("Endpoint Details:")
logger.info(f"  Name: {endpoint_details.name}")
logger.info(f"  State: {endpoint_details.state.ready}")
logger.info(f"  Creator: {endpoint_details.creator}")
logger.info(f"  Creation Time: {endpoint_details.creation_timestamp}")

if endpoint_details.config.served_entities:
    for entity in endpoint_details.config.served_entities:
        logger.info(f"\nServed Entity:")
        logger.info(f"  Model: {entity.entity_name}")
        logger.info(f"  Version: {entity.entity_version}")
        logger.info(f"  Workload Size: {entity.workload_size}")
        logger.info(f"  Scale to Zero: {entity.scale_to_zero_enabled}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Query Inference Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Inference Table

# COMMAND ----------

# Inference table name
inference_table = f"{cfg.catalog}.{cfg.schema}.arxiv_agent_{env}_payload"

logger.info(f"Inference table: {inference_table}")

# Query recent requests
recent_requests = spark.sql(f"""
    SELECT 
        request_id,
        timestamp,
        status_code,
        request,
        response
    FROM {inference_table}
    ORDER BY timestamp DESC
    LIMIT 10
""")

display(recent_requests)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Request Patterns

# COMMAND ----------

# Analyze request volume over time
request_stats = spark.sql(f"""
    SELECT 
        DATE_TRUNC('hour', timestamp) as hour,
        COUNT(*) as request_count,
        AVG(execution_duration_ms) as avg_latency_ms,
        MAX(execution_duration_ms) as max_latency_ms
    FROM {inference_table}
    WHERE timestamp >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
    GROUP BY hour
    ORDER BY hour DESC
""")

display(request_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ How to configure Model Serving endpoints
# MAGIC 2. ✅ How to deploy models from Unity Catalog
# MAGIC 3. ✅ How to enable inference tables for monitoring
# MAGIC 4. ✅ How to test deployed endpoints
# MAGIC 5. ✅ How to use the `arxiv_curator.serving` module
# MAGIC 6. ✅ How to query and analyze inference data
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Add guardrails and safety checks
# MAGIC - Set up evaluation pipelines
# MAGIC - Configure monitoring and alerts
# MAGIC - Implement A/B testing
