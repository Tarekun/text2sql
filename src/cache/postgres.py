from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import ProgrammingError
from src.logger import logger


Base = declarative_base()


DB_NAME = "postgres"
SCHEMA_NAME = "public"


def _pgvector_init(engine):
    with engine.connect() as conn:
        try:
            conn.execute(text(f"CREATE EXTENSION vector SCHEMA {SCHEMA_NAME}"))
            conn.commit()
            logger.debug("pgvector not found extension created...")
        except ProgrammingError as e:
            if 'extension "vector" already exists' in str(e):
                logger.debug(f"pgvector is already enabled")
            else:
                raise


def _tables_init(engine):
    logger.debug("Creating tables...")
    Base.metadata.create_all(engine)
    logger.debug("All tables created")


def get_engine():
    return create_engine(
        f"postgresql://admin:password@localhost:5432/{DB_NAME}",
        connect_args={"options": f"-csearch_path={SCHEMA_NAME},pg_catalog,pg_toast"},
    )


def initialization():
    engine = get_engine()
    _pgvector_init(engine)
    _tables_init(engine)

    logger.debug("Postgres instance for local data caching configured")
