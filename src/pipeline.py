# src/pipeline.py
"""
Main Data Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import io

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from .exceptions import (
    DataAnalysisError, ConfigurationError, DataLoadError,
    DataValidationError, AnalysisError
)
from .core.data_loader import DataLoader
from .core.validator import DataValidator
from .core.cleaner import DataCleaner
from .core.transformer import DataTransformer
from .core.feature_engineer import FeatureEngineer
from .analyzers.rfm_analyzer import RFMAnalyzer
from .analyzers.kpi_analyzer import KPIAnalyzer
from .analyzers.forecast_analyzer import ForecastAnalyzer
from .analyzers.seasonal_analyzer import SeasonalAnalyzer
from .analyzers.clv_analyzer import CLVAnalyzer
from .analyzers.retention_analyzer import RetentionAnalyzer
from .visualizer.visualizer import Visualizer
from .visualizer.dashboard import DashboardGenerator
from .reporter.reporter import Reporter
from .data_quality import DataQualityReporter
from .results_manager import ResultsManager


class DataPipeline:
    """Main Data Analysis Pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.results = {}
        self.quality_report = None
        self.data_columns = config.get('data', {}).get('columns', {})
        self._setup_logging()
        self._ensure_directories()
    
    def _setup_logging(self):
        """Setup logging from config"""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt=log_config.get('date_format', '%Y-%m-%d %H:%M:%S'),
            handlers=[
                logging.FileHandler(log_config.get('file', 'logs/pipeline.log'), encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = ['data', 'logs']
        output_config = self.config.get('output', {})
        directories.append(output_config.get('base_dir', 'outputs'))
        directories.append(output_config.get('reports_dir', 'outputs/reports'))
        directories.append(output_config.get('plots_dir', 'outputs/plots'))
        directories.append(output_config.get('data_dir', 'outputs/data'))
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING DATA ANALYSIS PIPELINE")
            self.logger.info("=" * 60)
            
            # Stage 1: Load data
            self.logger.info("Stage 1: Loading data...")
            self.data = self._load_data()
            
            # Stage 2: Validate data
            self.logger.info("Stage 2: Validating data...")
            self._validate_data()
            
            # Stage 3: Data Quality Report
            self.logger.info("Stage 3: Generating quality report...")
            self._generate_quality_report()
            
            # Stage 4: Clean data
            self.logger.info("Stage 4: Cleaning data...")
            self._clean_data()
            
            # Stage 5: Transform data
            self.logger.info("Stage 5: Transforming data...")
            self._transform_data()
            
            # Stage 6: Feature Engineering
            self.logger.info("Stage 6: Feature engineering...")
            self._engineer_features()
            
            # Stage 7: RFM Analysis
            self.logger.info("Stage 7: RFM analysis...")
            self.results['rfm'] = self._analyze_rfm()
            
            # Stage 8: KPIs
            self.logger.info("Stage 8: Calculating KPIs...")
            self.results['kpis'] = self._analyze_kpis()
            
            # Stage 9: Forecasting
            self.logger.info("Stage 9: Forecasting...")
            self.results['forecast'] = self._analyze_forecast()
            
            # Stage 10: Seasonal Analysis
            self.logger.info("Stage 10: Seasonal analysis...")
            self.results['seasonal'] = self._analyze_seasonal()
            
            # Stage 11: CLV Analysis
            self.logger.info("Stage 11: CLV analysis...")
            self.results['clv'] = self._analyze_clv()
            
            # Stage 12: Retention Analysis
            self.logger.info("Stage 12: Retention analysis...")
            self.results['retention'] = self._analyze_retention()
            
            # Stage 13: Churn Prediction (ML)
            self.logger.info("Stage 13: Churn prediction...")
            self.results['churn'] = self._predict_churn()
            
            # Stage 14: Generate visualizations
            self.logger.info("Stage 14: Generating visualizations...")
            self._generate_visualizations()
            
            # Stage 15: Generate dashboard
            self.logger.info("Stage 15: Generating dashboard...")
            self._generate_dashboard()
            
            # Stage 16: Generate reports
            self.logger.info("Stage 16: Generating reports...")
            self._generate_reports()
            
            # Stage 17: Save results
            self.logger.info("Stage 17: Saving results...")
            self._save_results()
            
            self.logger.info("=" * 60)
            self.logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            
            return self.results
            
        except DataAnalysisError as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise AnalysisError("Pipeline execution failed", details=str(e))
    
    def _load_data(self) -> pd.DataFrame:
        """Load data using configuration"""
        try:
            loader = DataLoader(self.config)
            return loader.load()
        except Exception as e:
            raise DataLoadError("Failed to load data", details=str(e))
    
    def _validate_data(self):
        """Validate data using configuration"""
        try:
            validator = DataValidator()
            expected_columns = list(self.data_columns.values()) if self.data_columns else ['customer_id', 'purchase_date', 'amount']
            validator.validate_schema(self.data, expected_columns)
            validator.check_missing_values(self.data)
            validator.check_outliers(self.data, self.data_columns.get('amount', 'amount'))
        except Exception as e:
            raise DataValidationError("Data validation failed", details=str(e))
    
    def _generate_quality_report(self):
        """Generate data quality report"""
        try:
            reporter = DataQualityReporter()
            self.quality_report = reporter.generate_report(self.data)
            self.results['quality_report'] = self.quality_report
            reporter.print_report()
        except Exception as e:
            self.logger.warning(f"Quality report generation failed: {str(e)}")
    
    def _clean_data(self):
        """Clean data"""
        try:
            cleaner = DataCleaner()
            cleaner.handle_missing(self.data)
            cleaner.remove_duplicates(self.data)
            date_col = self.data_columns.get('purchase_date', 'purchase_date')
            cleaner.standardize_dates(self.data, date_col)
        except Exception as e:
            raise DataValidationError("Data cleaning failed", details=str(e))
    
    def _transform_data(self):
        """Transform data"""
        try:
            transformer = DataTransformer()
            transformer.create_features(self.data)
        except Exception as e:
            raise AnalysisError("Data transformation failed", details=str(e))
    
    def _engineer_features(self):
        """Feature engineering"""
        try:
            engineer = FeatureEngineer(self.config)
            self.data = engineer.create_features(self.data)
        except Exception as e:
            raise AnalysisError("Feature engineering failed", details=str(e))
    
    def _analyze_rfm(self) -> Dict[str, Any]:
        """RFM Analysis"""
        try:
            analyzer = RFMAnalyzer()
            rfm_config = self.config.get('analysis', {}).get('rfm', {})
            reference_date = rfm_config.get('reference_date', None)
            
            rfm_df = analyzer.calculate_rfm(self.data, reference_date)
            rfm_segmented = analyzer.assign_segments(rfm_df)
            
            return {
                'data': rfm_segmented,
                'segment_counts': rfm_segmented['segment'].value_counts().to_dict(),
                'total_customers': len(rfm_segmented),
                'active_customers': len(rfm_segmented[rfm_segmented['recency'] <= 30])
            }
        except Exception as e:
            raise AnalysisError("RFM analysis failed", analysis_type='RFM', details=str(e))
    
    def _analyze_kpis(self) -> Dict[str, Any]:
        """Calculate KPIs"""
        try:
            analyzer = KPIAnalyzer()
            return analyzer.calculate(self.data)
        except Exception as e:
            raise AnalysisError("KPI calculation failed", analysis_type='KPIs', details=str(e))
    
    def _analyze_forecast(self) -> Dict[str, Any]:
        """Forecast"""
        try:
            analyzer = ForecastAnalyzer()
            periods = self.config.get('analysis', {}).get('forecast', {}).get('periods', 12)
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
    
    def _analyze_seasonal(self) -> Dict[str, Any]:
        """Seasonal Analysis"""
        try:
            analyzer = SeasonalAnalyzer()
            seasonal_config = self.config.get('analysis', {}).get('seasonal', {})
            period = seasonal_config.get('period', None)
            
            result = analyzer.decompose(self.data, period)
            result['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            return result
        except Exception as e:
            self.logger.error(f"Seasonal analysis error: {str(e)}")
            return {
                'method': 'error',
                'period': 7,
                'note': f'Error: {str(e)}',
                'seasonal_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'amplitude': 0}
            }
    
    def _analyze_clv(self) -> Dict[str, Any]:
        """CLV Analysis"""
        try:
            analyzer = CLVAnalyzer()
            return analyzer.calculate_clv(self.data)
        except Exception as e:
            raise AnalysisError("CLV analysis failed", analysis_type='CLV', details=str(e))
    
    def _analyze_retention(self) -> Dict[str, Any]:
        """Retention Analysis"""
        try:
            analyzer = RetentionAnalyzer()
            result = analyzer.cohort_analysis(self.data)
            result['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            return result
        except Exception as e:
            self.logger.error(f"Retention analysis error: {str(e)}")
            return {
                'summary': {
                    'total_customers': 0,
                    'avg_cohort_size': 0,
                    'num_cohorts': 0,
                    'max_period': 0
                },
                'error': str(e),
                'note': 'Retention analysis error'
            }
    
    def _predict_churn(self) -> Dict[str, Any]:
        """Churn Prediction"""
        try:
            from .ml.churn_predictor import ChurnPredictor
            predictor = ChurnPredictor(self.config)
            return predictor.predict(self.data)
        except Exception as e:
            self.logger.error(f"Churn prediction error: {str(e)}")
            return {
                'predictions': [],
                'summary': {
                    'total_customers': 0,
                    'avg_churn_probability': 0,
                    'high_risk_count': 0
                },
                'error': str(e)
            }
    
    def _generate_visualizations(self):
        """Generate visualizations"""
        try:
            visualizer = Visualizer()
            output_dir = self.config.get('visualization', {}).get('output_dir', 'outputs/plots')
            
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
    
    def _generate_dashboard(self):
        """Generate dashboard"""
        try:
            dashboard = DashboardGenerator(self.config)
            dashboard.generate(self.results)
            self.logger.info("Dashboard generated")
        except Exception as e:
            self.logger.error(f"Dashboard generation error: {str(e)}")
    
    def _generate_reports(self):
        """Generate reports"""
        try:
            reporter = Reporter()
            output_dir = self.config.get('output', {}).get('reports_dir', 'outputs/reports')
            
            # Excel report
            reporter.generate_excel_report(self.results, f"{output_dir}/report.xlsx")
            
            # HTML report
            reporter.generate_html_report(self.results, f"{output_dir}/report.html")
            
            self.logger.info("Reports generated")
            
        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")
    
    def _save_results(self):
        """Save results using ResultsManager"""
        try:
            manager = ResultsManager(self.config)
            saved_files = manager.save_results(self.results)
            
            self.logger.info("Results saved successfully:")
            for format_type, file_path in saved_files.items():
                self.logger.info(f"  - {format_type}: {file_path}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")