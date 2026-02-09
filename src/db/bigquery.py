from google.cloud import bigquery
from google.cloud.bigquery.table import Row
import re
import yaml


def _validate_query(query: str) -> str:
    for forbidden_keyword in ["INSERT", "ALTER", "UPDATE", "DROP", "DELETE"]:
        if forbidden_keyword in query:
            raise ValueError(
                f"Keyword {forbidden_keyword} is forbidden in this environment. DB altering statements have been disabled"
            )

    cleaned_query = query.removeprefix("```sql")
    cleaned_query = cleaned_query.removeprefix("```")
    cleaned_query = cleaned_query.removesuffix("```")
    cleaned_query = cleaned_query.strip().rstrip(";")
    return cleaned_query


def run_sql_query(query: str) -> list[Row]:
    query = _validate_query(query)
    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=100 * 1024 * 1024,  # 100 MB cap
    )
    client = bigquery.Client(project="soges-group-data-platform")
    # client = bigquery.Client(project="formazione-danieletarek-iaisy")
    query_job = client.query(query, job_config=job_config, timeout=30.0)
    result = query_job.result()

    return list(result), result.schema  # type:ignore


def gcp_pull_metadata(project_id: str, datasets: list[str] | None = None) -> list[dict]:
    """
    Fetches BigQuery metadata from GCP project and saves it to a YAML file.

    Args:
        project_id: GCP project ID
        datasets: list of dataset names to pull, if unspecified all datasets will be pulled
    Returns:
        A list of nested dictionaries containing metadata. The first level corresponds to datasets and contains:
        * name: name of the dataset
        * description: the description of the dataset available on BigQuery
        * others: other metadata found we dont currently have a use for
        * tables: list of dicts with metadata about tables in this dataset

        The second level corresponds to tables and contains keys:
        * name: name of the table
        * description: the description of the table available on BigQuery
        * others: other metadata found we dont currently have a use for
        * columns: list of dicts with metadata about columns of this table

        The third level corresponds to columns of tables and contains keys:
        * name: name of the column
        * description: the description of the column available on BigQuery
        * type: the data type of the column
    """

    client = bigquery.Client(project=project_id)
    metadata = []

    # dataset iteration to find available metadata
    for dataset_ref in client.list_datasets():
        dataset = client.get_dataset(dataset_ref.reference)
        if datasets is not None and dataset.dataset_id not in datasets:
            continue

        dataset_info = {
            "name": dataset.dataset_id,
            "kind": "dataset",
            "description": dataset.description or "",
            "tables": [],
            "others": _extract_other_metadata(dataset),
        }
        # getting tables metadata
        for table_ref in client.list_tables(dataset):
            table = client.get_table(table_ref.reference)
            table_info = {
                "name": table.table_id,
                "kind": "table",
                "description": table.description or "",
                "columns": [],
                "others": _extract_other_metadata(table),
            }

            # getting column metadata
            for field in table.schema:
                column_info = {
                    "name": field.name,
                    "type": field.field_type,
                    "description": field.description or "",
                }
                table_info["columns"].append(column_info)

            dataset_info["tables"].append(table_info)

        metadata.append(dataset_info)

    return metadata


def _extract_other_metadata(resource) -> dict:
    """
    Extracts additional metadata not covered by standard fields.
    """
    others = {}
    for attr in dir(resource):
        if attr in ["num_bytes", "num_rows", "table_type"]:
            try:
                value = getattr(resource, attr)
                if value is not None and not callable(value):
                    # Convert non-serializable types to strings
                    if isinstance(
                        value, (bigquery.SchemaField, bigquery.TableReference)
                    ):
                        continue
                    elif hasattr(value, "__dict__"):
                        others[attr] = str(value)
                    else:
                        others[attr] = value
            except Exception:
                # Skip attributes that can't be accessed
                pass
    return others
