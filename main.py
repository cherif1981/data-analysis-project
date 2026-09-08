# main.py - نسخة معدلة

import sys
import io
import os
import yaml
import pandas as pd
import logging
from pathlib import Path

# ✅ إعداد الترميز UTF-8 للـ CMD
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# إضافة مسار src إلى sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import DataPipeline

def setup_logging():
    """Setup logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('logs/main.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function"""
    logger = setup_logging()
    
    try:
        logger.info("=" * 60)
        logger.info("STARTING DATA ANALYSIS PROJECT")
        logger.info("=" * 60)
        
        # Check config file
        config_path = Path('config/config.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config
        logger.info("Loading configuration...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded successfully")
        logger.info(f"   - Data source: {config['data']['source']}")
        logger.info(f"   - Forecast periods: {config['analysis']['forecast']['periods']}")
        
        # Run pipeline
        logger.info("Running analysis pipeline...")
        pipeline = DataPipeline(config)
        results = pipeline.run()
        
        # Display summary
        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        if 'kpis' in results:
            kpis = results['kpis']
            logger.info(f"Total Revenue: ${kpis.get('total_revenue', 0):,.2f}")
            logger.info(f"Average Order Value: ${kpis.get('average_order_value', 0):,.2f}")
        
        if 'rfm' in results:
            rfm = results['rfm']
            if 'total_customers' in rfm:
                logger.info(f"Total Customers: {rfm['total_customers']}")
            if 'active_customers' in rfm:
                logger.info(f"Active Customers: {rfm['active_customers']}")
        
        if 'forecast' in results:
            forecast = results['forecast']
            logger.info(f"Forecast Method: {forecast.get('model', 'Unknown')}")
            if 'summary' in forecast:
                logger.info(f"Mean Forecast: ${forecast['summary'].get('mean_forecast', 0):,.2f}")
        
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info(f"Reports saved in 'reports/' folder")
        logger.info("=" * 60)
        
        return results
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()