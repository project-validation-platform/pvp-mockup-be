import io
import pandas as pd
from typing import Union
from pathlib import Path
import os
from urllib.parse import urlparse

def is_uri(s: str) -> bool:
    """
    Check if the object is a URI.
    
    Args:
        obj: The object to check.
        
    Returns:
        bool: True if the object is a URI, False otherwise.
    """
    if isinstance(s, str):
        return urlparse(s).scheme in ('http', 'https', 's3', 'gs', 'file')


def is_path(obj) -> bool:
    """
    Check if the object is a file path (string).
    
    Args:
        obj: The object to check.
        
    Returns:
        bool: True if the object is a string (file path), False otherwise.
    """

    # If it is a Path, return true
    if isinstance(obj, os.PathLike):
        return True
    
    # If it is string, check if it is a valid path
    if isinstance(obj, str):
        if is_uri(obj):
            return True
        return Path(obj).expanduser().exists()
    return False


def load_df(data: Union[str, bytes, io.BytesIO, pd.DataFrame], **read_csv_kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file path, bytes, or BytesIO object.
    
    Args:
        data (Union[str, bytes, io.BytesIO, pd.DataFrame]): The CSV file path or bytes-like object.
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if is_path(data):
        return pd.read_csv(data, **read_csv_kwargs)
    elif isinstance(data, str):
        return pd.read_csv(data, **read_csv_kwargs)
    elif isinstance(data, (bytes, io.BytesIO)):
        return pd.read_csv(io.BytesIO(data))
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        raise TypeError("Unsupported input. Provide a DataFrame, path/URL, bytes, BytesIO, or CSV text.")


