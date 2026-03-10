# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.3: arXiv Data Ingestion with Databricks Connect
# MAGIC ## Prerequisites:
# MAGIC - Databricks workspace with Unity Catalog enabled
# MAGIC - Unity Catalog: `llmops_dev.arxiv`

# COMMAND ----------
#%pip install ../arxiv_curator-0.1.0-py3-none-any.whl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Set Up Databricks Connect Session
# MAGIC
# MAGIC We'll use Databricks Connect to run this notebook locally and write to Unity Catalog.

# COMMAND ----------

from databricks.connect import DatabricksSession

# Create Databricks session
spark = DatabricksSession.builder.getOrCreate()

print("✅ Databricks Connect session created")
print(f"Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Unity Catalog
# MAGIC
# MAGIC Set up the catalog and schema for storing arXiv paper metadata.

# COMMAND ----------

from arxiv_curator.config import load_config, get_env

env = get_env()
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "arxiv_papers"

print(f"Unity Catalog: {CATALOG}.{SCHEMA}.{TABLE_NAME}")

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"✅ Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fetch arXiv Paper Metadata
# MAGIC
# MAGIC We'll fetch recent papers from arXiv in the AI/ML category.
# MAGIC
# MAGIC **arXiv API**: https://arxiv.org/help/api/index

# COMMAND ----------

import arxiv
from datetime import datetime

def fetch_arxiv_papers(query: str = "cat:cs.AI OR cat:cs.LG", max_results: int = 100):
    """
    Fetch arXiv papers using the arXiv API.
    
    Args:
        query: arXiv search query (default: AI and ML papers)
        max_results: Maximum number of papers to fetch
    
    Returns:
        List of paper metadata dictionaries
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in client.results(search):
        paper = {
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],  # Array to match reference code
            "summary": result.summary,
            "published": int(result.published.strftime("%Y%m%d%H%M")),  # Long to match reference code
            "updated": result.updated.isoformat() if result.updated else None,
            "categories": ", ".join(result.categories),
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "ingestion_timestamp": datetime.now().isoformat(),
            "processed": None,  # Will be set in Lecture 2.2
            "volume_path": None  # Will be set in Lecture 2.2
        }
        papers.append(paper)
    
    return papers

# Fetch papers
print("Fetching arXiv papers...")
papers = fetch_arxiv_papers(query="cat:cs.AI OR cat:cs.LG", max_results=50)
print(f"✅ Fetched {len(papers)} papers")

# Show sample
print("\nSample paper:")
print(f"Title: {papers[0]['title']}")
print(f"Authors: {papers[0]['authors']}")
print(f"arXiv ID: {papers[0]['arxiv_id']}")
print(f"PDF URL: {papers[0]['pdf_url']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Delta Table in Unity Catalog
# MAGIC
# MAGIC Store the arXiv paper metadata in a Delta table for downstream processing.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType

# Define schema
schema = StructType([
    StructField("arxiv_id", StringType(), False),
    StructField("title", StringType(), False),
    StructField("authors", ArrayType(StringType()), True),  # Array to match reference code
    StructField("summary", StringType(), True),
    StructField("published", LongType(), True),  # Long to match reference code
    StructField("updated", StringType(), True),
    StructField("categories", StringType(), True),
    StructField("pdf_url", StringType(), True),
    StructField("primary_category", StringType(), True),
    StructField("ingestion_timestamp", StringType(), True),
    StructField("processed", LongType(), True),  # Long to match reference code
    StructField("volume_path", StringType(), True)  # Will be set in Lecture 2.2
])

# Create DataFrame
df = spark.createDataFrame(papers, schema=schema)

# Write to Delta table
table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable(table_path)

print(f"✅ Created Delta table: {table_path}")
print(f"   Records: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify the Data

# COMMAND ----------

# Read back the table
papers_df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

print(f"Table: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
print(f"Total papers: {papers_df.count()}")
print("\nSchema:")
papers_df.printSchema()

print("\nSample records:")
papers_df.select("arxiv_id", "title", "primary_category", "published").show(5, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Statistics

# COMMAND ----------

# Category distribution
print("Papers by primary category:")
papers_df.groupBy("primary_category").count().orderBy("count", ascending=False).show()

# Recent papers
print("\nMost recent papers:")
papers_df.select("title", "published", "arxiv_id") \
    .orderBy("published", ascending=False) \
    .show(5, truncate=60)