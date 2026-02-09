from sqlalchemy import Column, Float, Integer, String
import yaml
from src.cache.postgres import Base, SCHEMA_NAME
from src.config import Config
from src.db.bigquery import gcp_pull_metadata


class DbMetadata(Base):
    __tablename__ = "db_metadata"
    __table_args__ = {"schema": SCHEMA_NAME}

    dataset = Column(String)
    dataset_description = Column(String)
    table = Column(String)
    table_description = Column(String)
    table_type = Column(String)
    table_bytes = Column(Integer)
    table_rows = Column(Integer)
    # columns = ???


def cache_all_metadata(config: Config):
    metadata = gcp_pull_metadata(config.gcp_project, ["gold"])
    with open("schema.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def get_table_metadata() -> str:
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

    return yaml_str
