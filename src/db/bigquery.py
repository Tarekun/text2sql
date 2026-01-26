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


def gcp_pull_metadata(project_id: str, datasets: list[str] | None = None) -> None:
    """
    Fetches BigQuery metadata from GCP project and saves it to a YAML file.

    Args:
        project_id: GCP project ID
        datasets: list of dataset names to pull, if unspecified all datasets will be pulled
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

    with open("schema.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


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


def get_table_metadata():
    with open("schema.yaml", "r") as f:
        yaml_str = f.read()
        datasets = yaml.safe_load(yaml_str)

    if not datasets:
        return "No datasets found in schema file."

    schema_str = ""
    for dataset in datasets:
        dataset_name = dataset["name"]

        for table in dataset.get("tables", []):
            full_table_name = f"{dataset_name}.{table['name']}"
            table_desc = table.get("description", "(Description not available)")
            schema_str += f"Table: {full_table_name}\n"
            schema_str += f"\tDescription: {table_desc}\n"
            schema_str += f"\tByte usage: {table['others']['num_bytes']}\n"
            schema_str += "\tColumns:\n"

            for column in table.get("columns", []):
                col_name = column["name"]
                col_type = column["type"]
                col_desc = column.get("description", "(Description not available)")
                schema_str += f"\t\t{col_name}:{col_type} {col_desc}\n"

            schema_str += "\n"

    # return schema_str
    return yaml_str
