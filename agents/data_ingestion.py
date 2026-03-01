"""
Agent 1: Data Ingestion
=======================
Load datasets from CSV, Excel, Parquet files or URLs (table scraping).
No LLM calls - pure Python/pandas logic.
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Optional
from io import StringIO

from pipeline_context import PipelineContext
from config import MAX_ROWS
from logger import get_logger

logger = get_logger("DataIngestion")

SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.parquet', '.pq'}


def _detect_format(path: str) -> str:
    """Detect file format from extension or URL."""
    if path.startswith('http://') or path.startswith('https://'):
        return 'url'
    
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    return 'unsupported'


def _load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file with row limit."""
    logger.info(f"Loading CSV file: {filepath}")
    try:
        df = pd.read_csv(filepath, nrows=MAX_ROWS)
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")


def _load_excel(filepath: str) -> pd.DataFrame:
    """Load Excel file (.xlsx or .xls) with row limit."""
    logger.info(f"Loading Excel file: {filepath}")
    try:
        df = pd.read_excel(filepath, nrows=MAX_ROWS)
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse Excel file: {str(e)}")


def _load_parquet(filepath: str) -> pd.DataFrame:
    """Load Parquet file with row limit applied after loading."""
    logger.info(f"Loading Parquet file: {filepath}")
    try:
        df = pd.read_parquet(filepath)
        if len(df) > MAX_ROWS:
            logger.warning(f"Dataset exceeds MAX_ROWS ({MAX_ROWS}). Truncating from {len(df)} rows.")
            df = df.head(MAX_ROWS)
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse Parquet file: {str(e)}")


def _scrape_tables_from_url(url: str) -> pd.DataFrame:
    """Scrape tables from URL, return the largest table by row count."""
    logger.info(f"Scraping tables from URL: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Connection error: Unable to reach {url}")
    except requests.exceptions.Timeout:
        raise ValueError(f"Timeout error: Request to {url} timed out")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to fetch URL: {str(e)}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')
    
    if not tables:
        raise ValueError(f"No tables found at URL: {url}")
    
    # Convert each table to DataFrame and find the largest
    dataframes = []
    for i, table in enumerate(tables):
        try:
            # Use pd.read_html with StringIO to parse table HTML
            table_html = str(table)
            dfs = pd.read_html(StringIO(table_html))
            if dfs:
                dataframes.append(dfs[0])
        except Exception as e:
            logger.debug(f"Could not parse table {i}: {str(e)}")
            continue
    
    if not dataframes:
        raise ValueError(f"No parseable tables found at URL: {url}")
    
    # Find the largest table by row count
    largest_df = max(dataframes, key=lambda df: len(df))
    
    if len(dataframes) > 1:
        logger.warning(f"Multiple tables found ({len(dataframes)}), using largest ({len(largest_df)} rows)")
    
    # Apply MAX_ROWS limit
    if len(largest_df) > MAX_ROWS:
        logger.warning(f"Dataset exceeds MAX_ROWS ({MAX_ROWS}). Truncating from {len(largest_df)} rows.")
        largest_df = largest_df.head(MAX_ROWS)
    
    return largest_df


def run_data_ingestion(ctx: PipelineContext) -> PipelineContext:
    """
    Load dataset from file path or URL into ctx.raw_df.
    Supports CSV, Excel, Parquet, and URL table scraping.
    
    Args:
        ctx: PipelineContext with dataset_path set
        
    Returns:
        Updated PipelineContext with raw_df populated
    """
    ctx.mark_agent("Data Ingestion", "running")
    logger.info("=" * 50)
    logger.info("Starting Data Ingestion Agent")
    logger.info("=" * 50)
    
    path = ctx.dataset_path
    
    if not path:
        error_msg = "No dataset path provided"
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("Data Ingestion", "failed")
        return ctx
    
    logger.info(f"Dataset path: {path}")
    
    try:
        # Detect format
        fmt = _detect_format(path)
        logger.info(f"Detected format: {fmt}")
        
        if fmt == 'unsupported':
            raise ValueError(f"Unsupported file format. Supported: {', '.join(SUPPORTED_EXTENSIONS)} or URL")
        
        # Check file exists (for local files)
        if fmt != 'url' and not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load data based on format
        if fmt == '.csv':
            df = _load_csv(path)
            ctx.dataset_name = os.path.basename(path)
        elif fmt in ['.xlsx', '.xls']:
            df = _load_excel(path)
            ctx.dataset_name = os.path.basename(path)
        elif fmt in ['.parquet', '.pq']:
            df = _load_parquet(path)
            ctx.dataset_name = os.path.basename(path)
        elif fmt == 'url':
            df = _scrape_tables_from_url(path)
            ctx.dataset_name = "scraped_data"
        
        # Validate loaded data
        if df is None or df.empty:
            raise ValueError("Dataset is empty after loading")
        
        # Check row count and apply limit if needed (for CSV/Excel which use nrows)
        original_rows = len(df)
        if original_rows > MAX_ROWS:
            logger.warning(f"Dataset exceeds MAX_ROWS ({MAX_ROWS}). Truncating from {original_rows} rows.")
            df = df.head(MAX_ROWS)
        
        # Store results in context
        ctx.raw_df = df
        
        logger.info(f"Successfully loaded dataset: {ctx.dataset_name}")
        logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        ctx.mark_agent("Data Ingestion", "done")
        logger.info("Data Ingestion Agent completed successfully")
        
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("Data Ingestion", "failed")
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("Data Ingestion", "failed")
        
    except Exception as e:
        error_msg = f"Unexpected error during data ingestion: {str(e)}"
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("Data Ingestion", "failed")
    
    return ctx
