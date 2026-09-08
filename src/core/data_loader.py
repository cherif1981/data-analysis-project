# src/core/data_loader.py
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional

from ..exceptions import DataLoadError

class DataLoader:
    """Load data from various sources using configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_config = config.get('data', {})
    
    def load(self) -> pd.DataFrame:
        """Load data based on configuration"""
        source = self.data_config.get('source')
        
        if not source:
            raise DataLoadError("Data source not specified in config")
        
        try:
            self.logger.info(f"Loading data from: {source}")
            
            if source.endswith('.csv'):
                return self._load_csv(source)
            elif source.endswith(('.xlsx', '.xls')):
                return self._load_excel(source)
            else:
                raise DataLoadError(f"Unsupported file type: {source}")
                
        except Exception as e:
            raise DataLoadError(f"Failed to load data from {source}", filepath=source, details=str(e))
    
    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV with config options"""
        try:
            path = Path(filepath)
            if not path.exists():
                # Try alternative paths
                alt_path = Path('data') / path.name
                if alt_path.exists():
                    path = alt_path
                else:
                    raise DataLoadError(f"File not found: {filepath}", filepath=filepath)
            
            df = pd.read_csv(
                path,
                encoding=self.data_config.get('encoding', 'utf-8'),
                parse_dates=[self.data_config.get('columns', {}).get('purchase_date', 'purchase_date')] if self.data_config.get('parse_dates', True) else None
            )
            
            self.logger.info(f"Loaded {len(df)} records from {path}")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV", filepath=filepath, details=str(e))
    
    def _load_excel(self, filepath: str) -> pd.DataFrame:
        """Load Excel with config options"""
        try:
            path = Path(filepath)
            if not path.exists():
                raise DataLoadError(f"File not found: {filepath}", filepath=filepath)
            
            df = pd.read_excel(path)
            self.logger.info(f"Loaded {len(df)} records from {path}")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load Excel", filepath=filepath, details=str(e))