# src/pipeline.py - نسخة معدلة بدون أحرف عربية في الرسائل

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import sys
import io

# ✅ إعداد الترميز
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from .core.data_loader import DataLoader
from .core.validator import DataValidator
from .core.cleaner import DataCleaner
from .core.transformer import DataTransformer
from .analyzers.rfm_analyzer import RFMAnalyzer
from .analyzers.kpi_analyzer import KPIAnalyzer
from .analyzers.forecast_analyzer import ForecastAnalyzer
from .analyzers.seasonal_analyzer import SeasonalAnalyzer
from .analyzers.clv_analyzer import CLVAnalyzer
from .analyzers.retention_analyzer import RetentionAnalyzer
from .visualizer.visualizer import Visualizer
from .reporter.reporter import Reporter

class DataPipeline:
    """Data Analysis Pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.results = {}
        self.logger = self._setup_logging()
        self._ensure_directories()
    
    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = ['data', 'reports', 'reports/plots', 'logs', 'config']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("Starting Data Analysis Pipeline")
            self.logger.info("=" * 50)
            
            # 1. Load data
            self.logger.info("Loading data...")
            self.data = self._load_data()
            
            # 2. Validate data
            self.logger.info("Validating data...")
            self._validate_data()
            
            # 3. Clean data
            self.logger.info("Cleaning data...")
            self._clean_data()
            
            # 4. Transform data
            self.logger.info("Transforming data...")
            self._transform_data()
            
            # 5. RFM Analysis
            self.logger.info("Running RFM analysis...")
            self.results['rfm'] = self._analyze_rfm()
            
            # 6. KPIs
            self.logger.info("Calculating KPIs...")
            self.results['kpis'] = self._analyze_kpis()
            
            # 7. Forecast
            self.logger.info("Running forecast...")
            self.results['forecast'] = self._analyze_forecast()
            
            # 8. Seasonal Analysis
            self.logger.info("Running seasonal analysis...")
            self.results['seasonal'] = self._analyze_seasonal()
            
            # 9. CLV Analysis
            self.logger.info("Running CLV analysis...")
            self.results['clv'] = self._analyze_clv()
            
            # 10. Retention Analysis
            self.logger.info("Running retention analysis...")
            self.results['retention'] = self._analyze_retention()
            
            # 11. Generate visualizations
            self.logger.info("Generating visualizations...")
            self._generate_visualizations()
            
            # 12. Generate reports
            self.logger.info("Generating reports...")
            self._generate_reports()
            
            self.logger.info("=" * 50)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("=" * 50)
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            raise
    
    def _load_data(self):
        """Load data from source"""
        loader = DataLoader()
        source = self.config['data']['source']
        
        if source.endswith('.csv'):
            return loader.load_csv(source)
        elif source.endswith('.xlsx'):
            return loader.load_excel(source)
        else:
            raise ValueError(f"Unsupported file type: {source}")
    
    def _validate_data(self):
        """Validate data"""
        validator = DataValidator()
        expected_columns = ['customer_id', 'purchase_date', 'amount']
        validator.validate_schema(self.data, expected_columns)
        validator.check_missing_values(self.data)
        validator.check_outliers(self.data, 'amount')
    
    def _clean_data(self):
        """Clean data"""
        cleaner = DataCleaner()
        cleaner.handle_missing(self.data)
        cleaner.remove_duplicates(self.data)
        cleaner.standardize_dates(self.data, self.config['data']['date_column'])
    
    def _transform_data(self):
        """Transform data"""
        transformer = DataTransformer()
        transformer.create_features(self.data)
    
    def _analyze_rfm(self):
        """RFM Analysis"""
        try:
            analyzer = RFMAnalyzer()
            reference_date = self.config.get('analysis', {}).get('rfm', {}).get('reference_date', None)
            rfm_df = analyzer.calculate_rfm(self.data, reference_date)
            rfm_segmented = analyzer.assign_segments(rfm_df)
            
            segment_recommendations = {}
            for segment in rfm_segmented['segment'].unique():
                segment_recommendations[segment] = analyzer.get_segment_recommendations(segment)
            
            result = {
                'data': rfm_segmented,
                'segment_counts': rfm_segmented['segment'].value_counts().to_dict(),
                'segment_stats': rfm_segmented.groupby('segment').agg({
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': 'mean'
                }).to_dict(),
                'recommendations': segment_recommendations,
                'total_customers': len(rfm_segmented),
                'active_customers': len(rfm_segmented[rfm_segmented['recency'] <= 30])
            }
            
            self.logger.info(f"RFM analysis complete: {result['total_customers']} customers")
            return result
            
        except Exception as e:
            self.logger.error(f"RFM analysis error: {str(e)}")
            raise
    
    def _analyze_kpis(self):
        """Calculate KPIs"""
        analyzer = KPIAnalyzer()
        return analyzer.calculate(self.data)
    
    def _analyze_forecast(self):
        """Forecast"""
        try:
            analyzer = ForecastAnalyzer()
            periods = self.config['analysis']['forecast']['periods']
            result = analyzer.forecast(self.data, periods)
            self.logger.info(f"Forecast complete using {result.get('model', 'unknown')}")
            return result
        except Exception as e:
            self.logger.error(f"Forecast error: {str(e)}")
            return {
                'model': 'Error',
                'periods': 0,
                'forecast': pd.DataFrame(),
                'summary': {'mean_forecast': 0, 'total_forecast': 0},
                'error': str(e)
            }
    
    def _analyze_seasonal(self):
        """Seasonal Analysis"""
        try:
            from .analyzers.seasonal_analyzer import SeasonalAnalyzer
            analyzer = SeasonalAnalyzer()
            period = self.config.get('analysis', {}).get('seasonal', {}).get('period', None)
            result = analyzer.decompose(self.data, period)
            
            if result:
                result['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                result['data_points'] = len(self.data)
                self.logger.info(f"Seasonal analysis complete using {result.get('method', 'unknown')}")
                return result
            else:
                return {
                    'method': 'error',
                    'period': 7,
                    'note': 'Seasonal analysis failed',
                    'seasonal_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'amplitude': 0}
                }
        except Exception as e:
            self.logger.error(f"Seasonal analysis error: {str(e)}")
            return {
                'method': 'error',
                'period': 7,
                'note': f'Error: {str(e)}',
                'seasonal_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'amplitude': 0}
            }
    
    def _analyze_clv(self):
        """CLV Analysis"""
        analyzer = CLVAnalyzer()
        return analyzer.calculate_clv(self.data)
    
    def _analyze_retention(self):
        """Retention Analysis"""
        try:
            from .analyzers.retention_analyzer import RetentionAnalyzer
            analyzer = RetentionAnalyzer()
            result = analyzer.cohort_analysis(self.data)
            
            if result:
                result['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                result['data_points'] = len(self.data)
                
                if 'summary' in result:
                    self.logger.info(f"Retention analysis complete: {result['summary']['num_cohorts']} cohorts")
                else:
                    self.logger.info("Retention analysis complete")
                return result
            else:
                return {
                    'summary': {
                        'total_customers': 0,
                        'avg_cohort_size': 0,
                        'num_cohorts': 0,
                        'max_period': 0,
                        'avg_retention': {}
                    },
                    'note': 'Retention analysis failed',
                    'error': 'Empty result'
                }
        except Exception as e:
            self.logger.error(f"Retention analysis error: {str(e)}")
            return {
                'summary': {
                    'total_customers': 0,
                    'avg_cohort_size': 0,
                    'num_cohorts': 0,
                    'max_period': 0,
                    'avg_retention': {}
                },
                'error': str(e),
                'note': 'Retention analysis error'
            }
    
    def _generate_visualizations(self):
        """Generate visualizations"""
        try:
            visualizer = Visualizer()
            output_dir = self.config['visualization']['output_dir']
            
            if 'rfm' in self.results and 'data' in self.results['rfm']:
                visualizer.plot_rfm_distribution(self.results['rfm']['data'])
            
            if 'forecast' in self.results:
                visualizer.plot_forecast(self.results['forecast'])
            
            if 'seasonal' in self.results:
                visualizer.plot_seasonal_decomposition(self.results['seasonal'])
            
            if 'retention' in self.results:
                visualizer.plot_cohort_retention(self.results['retention'])
            
            visualizer.save_plots(output_dir)
            self.logger.info("Visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}")
    
    def _generate_reports(self):
        """Generate reports"""
        try:
            reporter = Reporter()
            output_dir = self.config['reporting']['output_dir']
            formats = self.config['reporting']['formats']
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            for fmt in formats:
                try:
                    if fmt == 'excel':
                        self.logger.info("Creating Excel report...")
                        reporter.generate_excel_report(
                            self.results, 
                            f"{output_dir}/report.xlsx"
                        )
                    elif fmt == 'html':
                        self.logger.info("Creating HTML report...")
                        reporter.generate_html_report(
                            self.results, 
                            f"{output_dir}/report.html"
                        )
                    elif fmt == 'pdf':
                        self.logger.info("Creating PDF report...")
                        reporter.generate_pdf_report(
                            self.results, 
                            f"{output_dir}/report.pdf"
                        )
                    else:
                        self.logger.warning(f"Unknown format: {fmt}")
                except Exception as e:
                    self.logger.error(f"Error creating {fmt} report: {str(e)}")
                    continue
            
            self.logger.info("All reports generated")
            
        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")