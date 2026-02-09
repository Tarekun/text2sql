from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, JSON, PrimaryKeyConstraint
from sqlalchemy.orm import Session
import yaml
from src.cache.embeddings import embed_text
from src.cache.postgres import Base, SCHEMA_NAME, get_engine
from src.config import Config
from src.db.bigquery import gcp_pull_metadata
from src.logger import logger


class DbMetadata(Base):
    __tablename__ = "db_metadata"
    __table_args__ = (PrimaryKeyConstraint("dataset", "table"), {"schema": SCHEMA_NAME})

    dataset = Column(String, nullable=False)
    dataset_description = Column(String)
    table = Column(String, nullable=False)
    table_description = Column(String)
    # table/view/materialized/...
    table_type = Column(String)
    table_bytes = Column(Integer)
    table_rows = Column(Integer)
    # JSON array of objects with keys: name, type, description
    columns = Column(JSON)
    embedding = Column(Vector(768), nullable=True)


def cache_all_metadata(config: Config):
    metadata = gcp_pull_metadata(config.gcp_project, ["gold"])
    with open("schema.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    engine = get_engine()
    with Session(engine) as session:
        # Clear existing metadata
        # session.query(DbMetadata).delete()
        for dataset in metadata:
            dataset_name = dataset.get("name")
            dataset_description = dataset.get("description", "")
            for table in dataset.get("tables", []):
                table_name = table.get("name")
                table_description = table.get("description", "")
                table_kind = table.get("kind", "table")
                table_type = table_kind.upper()
                # Extract table statistics
                others = table.get("others", {})
                table_bytes = others.get("num_bytes")
                table_rows = others.get("num_rows")
                # Extract columns as JSON-compatible list
                columns_data = []
                for column in table.get("columns", []):
                    columns_data.append(
                        {
                            "name": column.get("name"),
                            "type": column.get("type"),
                            "description": column.get("description", ""),
                        }
                    )

                embedding = None
                if config.embed_metadata:
                    schema = ",\n".join(
                        [
                            f"{c['name']} : {c['type']} ({c['description']})"
                            for c in columns_data
                        ]
                    )
                    table_text = f"Entity({table_type}) {dataset_name}.{table_name}\nDescription: {table_description}\nSchema:{schema}"
                    embedding = embed_text(
                        table_text,
                        model=config.embedding_model,
                        api_base=config.embedding_api_base,
                    )

                db_metadata = DbMetadata(
                    dataset=dataset_name,
                    dataset_description=dataset_description,
                    table=table_name,
                    table_description=table_description,
                    table_type=table_type,
                    table_bytes=table_bytes,
                    table_rows=table_rows,
                    columns=columns_data,
                    embedding=embedding,
                )
                session.add(db_metadata)

        session.commit()

    logger.debug("Fully dumped database metadata")


def get_table_metadata() -> str:
    engine = get_engine()
    with Session(engine) as session:
        metadata_records = session.query(DbMetadata).all()
        if not metadata_records:
            return "No datasets found in metadata database."

        schema_str = ""
        for record in metadata_records:
            full_table_name = f"{record.dataset}.{record.table}"
            table_desc = record.table_description or "(Description not available)"
            schema_str += f"Table: {full_table_name}\n"
            schema_str += f"\tDescription: {table_desc}\n"
            schema_str += f"\tByte usage: {record.table_bytes}\n"
            schema_str += "\tColumns:\n"

            columns = record.columns or []
            for column in columns:
                col_name = column["name"]
                col_type = column["type"]
                col_desc = (
                    column.get("description", "") or "(Description not available)"
                )
                schema_str += f"\t\t{col_name}:{col_type} {col_desc}\n"

            schema_str += "\n"

        return schema_str
