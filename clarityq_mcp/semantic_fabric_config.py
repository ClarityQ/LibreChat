import os

API_URL = os.environ.get("SEMANTIC_FABRIC_API_URL", "http://localhost:8000/api/v1/")
AUTH0_CLIENT_ID = os.environ.get("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.environ.get("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN")
PRODUCT_ID = os.environ.get("PRODUCT_ID")
DATA_WAREHOUSE_TYPE = os.environ.get("DATA_WAREHOUSE_TYPE", "BigQuery")
