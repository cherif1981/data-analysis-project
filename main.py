# main.py
import sys
import io
import yaml
from pathlib import Path
import logging

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.pipeline import DataPipeline
from src.exceptions import ConfigurationError

def main():
    """Main function"""
    try:
        print("=" * 60)
        print("DATA ANALYSIS PROJECT")
        print("=" * 60)
        
        # Load configuration
        config_path = Path('config/config.yaml')
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        print("Loading configuration...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Run pipeline
        print("Running analysis pipeline...")
        pipeline = DataPipeline(config)
        results = pipeline.run()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display summary
        if 'kpis' in results:
            kpis = results['kpis']
            print(f"\n📊 KPIs Summary:")
            print(f"   Total Revenue: ${kpis.get('total_revenue', 0):,.2f}")
            print(f"   Average Order Value: ${kpis.get('average_order_value', 0):,.2f}")
        
        if 'rfm' in results:
            rfm = results['rfm']
            print(f"\n👥 Customer Summary:")
            print(f"   Total Customers: {rfm.get('total_customers', 0)}")
            print(f"   Active Customers: {rfm.get('active_customers', 0)}")
        
        if 'forecast' in results:
            forecast = results['forecast']
            print(f"\n🔮 Forecast Summary:")
            print(f"   Method: {forecast.get('model', 'Unknown')}")
            if 'summary' in forecast:
                print(f"   Mean Forecast: ${forecast['summary'].get('mean_forecast', 0):,.2f}")
        
        print("\n📁 Output files saved in 'outputs/' directory")
        print("=" * 60)
        
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()