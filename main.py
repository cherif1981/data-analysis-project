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
        print("DATA ANALYSIS PROJECT - PRODUCTION PIPELINE")
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
        if results:
            print(f"\n📊 Pipeline Summary:")
            print(f"   Total Samples: {results.get('total_samples', 0)}")
            print(f"   Training Samples: {results.get('train_size', 0)}")
            print(f"   Validation Samples: {results.get('val_size', 0)}")
            print(f"   Test Samples: {results.get('test_size', 0)}")
            print(f"   Model Performance: {results.get('model_performance', 'N/A')}")
        
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
