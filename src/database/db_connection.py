import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def get_database_url() -> str:
    """Build PostgreSQL SQLAlchemy connection URL from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "retailpulse_db")
    user = os.getenv("DB_USER", "postgres")
    password = quote_plus(os.getenv("DB_PASSWORD", ""))
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"


def get_engine() -> Engine:
    """Create a pooled SQLAlchemy engine."""
    return create_engine(get_database_url(), pool_pre_ping=True)


def verify_connection(engine: Engine) -> bool:
    """Verify database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection verified successfully.")
        return True
    except Exception as exc:
        logger.error(f"PostgreSQL connection failed: {exc}")
        raise
