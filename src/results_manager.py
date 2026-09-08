# src/results_manager.py
"""
Manage and save analysis results in multiple formats
"""

import pandas as pd
import json
import csv
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np

from .exceptions import ReportGenerationError

class ResultsManager:
    """Save analysis results in various formats"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_config = config.get('output', {})
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all required output directories"""
        base_dir = self.output_config.get('base_dir', 'outputs')
        directories = [
            base_dir,
            self.output_config.get('reports_dir', f'{base_dir}/reports'),
            self.output_config.get('plots_dir', f'{base_dir}/plots'),
            self.output_config.get('data_dir', f'{base_dir}/data')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")
    
    def save_results(self, results: Dict[str, Any], timestamp: bool = True) -> Dict[str, str]:
        """
        Save all results in configured formats
        
        Args:
            results: Dictionary containing analysis results
            timestamp: Whether to add timestamp to file names
            
        Returns:
            Dictionary with saved file paths
        """
        saved_files = {}
        base_dir = self.output_config.get('base_dir', 'outputs')
        
        # Get output formats
        formats = self.output_config.get('formats', {})
        file_names = self.output_config.get('file_names', {})
        
        # Add timestamp if enabled
        time_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
        
        try:
            # 1. Save CSV data
            if formats.get('csv', True):
                csv_file = self._save_csv(results, base_dir, file_names, time_suffix)
                saved_files['csv'] = csv_file
            
            # 2. Save JSON summary
            if formats.get('json', True):
                json_file = self._save_json(results, base_dir, file_names, time_suffix)
                saved_files['json'] = json_file
            
            # 3. Save Excel report
            if formats.get('excel', True):
                excel_file = self._save_excel(results, base_dir, file_names, time_suffix)
                saved_files['excel'] = excel_file
            
            # 4. Save HTML report
            if formats.get('html', True):
                html_file = self._save_html(results, base_dir, file_names, time_suffix)
                saved_files['html'] = html_file
            
            # 5. Save PDF (optional)
            if formats.get('pdf', False):
                pdf_file = self._save_pdf(results, base_dir, file_names, time_suffix)
                saved_files['pdf'] = pdf_file
            
            # 6. Save visualizations (PNG)
            if formats.get('visualizations', True):
                plots_dir = self.output_config.get('plots_dir', f'{base_dir}/plots')
                self._save_visualizations(results, plots_dir, time_suffix)
            
            self.logger.info(f"✅ All results saved successfully")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise ReportGenerationError("Failed to save results", details=str(e))
    
    def _save_csv(self, results: Dict[str, Any], base_dir: str, file_names: Dict, suffix: str) -> str:
        """Save data as CSV files"""
        try:
            data_dir = self.output_config.get('data_dir', f'{base_dir}/data')
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # Save main data
            if 'data' in results and isinstance(results['data'], pd.DataFrame):
                file_name = f"data_{suffix}.csv" if suffix else "data.csv"
                file_path = Path(data_dir) / file_name
                results['data'].to_csv(file_path, index=False, encoding='utf-8-sig')
                saved_files.append(str(file_path))
                self.logger.info(f"Saved CSV data: {file_path}")
            
            # Save KPIs as CSV
            if 'kpis' in results and isinstance(results['kpis'], dict):
                file_name = f"kpis_{suffix}.csv" if suffix else "kpis.csv"
                file_path = Path(data_dir) / file_name
                kpis_df = pd.DataFrame([results['kpis']])
                kpis_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                saved_files.append(str(file_path))
                self.logger.info(f"Saved KPIs CSV: {file_path}")
            
            # Save RFM data
            if 'rfm' in results and 'data' in results['rfm']:
                file_name = f"rfm_data_{suffix}.csv" if suffix else "rfm_data.csv"
                file_path = Path(data_dir) / file_name
                results['rfm']['data'].to_csv(file_path, index=False, encoding='utf-8-sig')
                saved_files.append(str(file_path))
                self.logger.info(f"Saved RFM data CSV: {file_path}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Failed to save CSV: {str(e)}")
            raise
    
    def _save_json(self, results: Dict[str, Any], base_dir: str, file_names: Dict, suffix: str) -> str:
        """Save summary as JSON"""
        try:
            data_dir = self.output_config.get('data_dir', f'{base_dir}/data')
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            
            file_name = file_names.get('json_summary', 'summary.json')
            if suffix:
                file_name = file_name.replace('.json', f'{suffix}.json')
            
            file_path = Path(data_dir) / file_name
            
            # Convert DataFrames to dict for JSON
            json_data = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    json_data[key] = value.to_dict(orient='records')
                elif isinstance(value, dict):
                    # Handle nested DataFrames
                    json_data[key] = {}
                    for k, v in value.items():
                        if isinstance(v, pd.DataFrame):
                            json_data[key][k] = v.to_dict(orient='records')
                        else:
                            json_data[key][k] = v
                else:
                    json_data[key] = value
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Saved JSON summary: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {str(e)}")
            raise
    
    def _save_excel(self, results: Dict[str, Any], base_dir: str, file_names: Dict, suffix: str) -> str:
        """Save Excel report with multiple sheets"""
        try:
            reports_dir = self.output_config.get('reports_dir', f'{base_dir}/reports')
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            
            file_name = file_names.get('excel_report', 'report.xlsx')
            if suffix:
                file_name = file_name.replace('.xlsx', f'{suffix}.xlsx')
            
            file_path = Path(reports_dir) / file_name
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = self._create_summary(results)
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # KPIs sheet
                if 'kpis' in results and isinstance(results['kpis'], dict):
                    kpis_df = pd.DataFrame([results['kpis']])
                    kpis_df.to_excel(writer, sheet_name='KPIs', index=False)
                
                # RFM sheet
                if 'rfm' in results and 'data' in results['rfm']:
                    results['rfm']['data'].to_excel(writer, sheet_name='RFM', index=False)
                
                # Retention matrix
                if 'retention' in results and 'retention_rates' in results['retention']:
                    results['retention']['retention_rates'].to_excel(writer, sheet_name='Retention')
                
                # Forecast
                if 'forecast' in results and 'forecast' in results['forecast']:
                    results['forecast']['forecast'].to_excel(writer, sheet_name='Forecast', index=False)
            
            self.logger.info(f"Saved Excel report: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save Excel: {str(e)}")
            raise
    
    def _save_html(self, results: Dict[str, Any], base_dir: str, file_names: Dict, suffix: str) -> str:
        """Save HTML report"""
        try:
            reports_dir = self.output_config.get('reports_dir', f'{base_dir}/reports')
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            
            file_name = file_names.get('html_report', 'report.html')
            if suffix:
                file_name = file_name.replace('.html', f'{suffix}.html')
            
            file_path = Path(reports_dir) / file_name
            
            html_content = self._create_html_content(results)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Saved HTML report: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save HTML: {str(e)}")
            raise
    
    def _save_pdf(self, results: Dict[str, Any], base_dir: str, file_names: Dict, suffix: str) -> Optional[str]:
        """Save PDF report (optional)"""
        try:
            # PDF is optional - skip if weasyprint not available
            try:
                from weasyprint import HTML
            except ImportError:
                self.logger.warning("WeasyPrint not available, skipping PDF")
                return None
            
            reports_dir = self.output_config.get('reports_dir', f'{base_dir}/reports')
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            
            file_name = file_names.get('pdf_report', 'report.pdf')
            if suffix:
                file_name = file_name.replace('.pdf', f'{suffix}.pdf')
            
            file_path = Path(reports_dir) / file_name
            
            # Create HTML then convert to PDF
            html_content = self._create_html_content(results)
            HTML(string=html_content).write_pdf(file_path)
            
            self.logger.info(f"Saved PDF report: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to save PDF: {str(e)}")
            return None
    
    def _save_visualizations(self, results: Dict[str, Any], plots_dir: str, suffix: str):
        """Save visualizations as PNG files"""
        try:
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            
            # Try to create visualizations if visualizer is available
            try:
                from .visualizer.visualizer import Visualizer
                visualizer = Visualizer()
                
                # Save RFM plot
                if 'rfm' in results and 'data' in results['rfm']:
                    file_name = f"rfm_distribution{suffix}.png"
                    file_path = Path(plots_dir) / file_name
                    visualizer.plot_rfm_distribution(results['rfm']['data'])
                    # Rename the saved file
                    self._rename_plot('reports/plots/rfm_distribution.png', file_path)
                
                # Save forecast plot
                if 'forecast' in results:
                    file_name = f"forecast{suffix}.png"
                    file_path = Path(plots_dir) / file_name
                    visualizer.plot_forecast(results['forecast'])
                    self._rename_plot('reports/plots/forecast.png', file_path)
                
                # Save retention plot
                if 'retention' in results:
                    file_name = f"retention_matrix{suffix}.png"
                    file_path = Path(plots_dir) / file_name
                    visualizer.plot_cohort_retention(results['retention'])
                    self._rename_plot('reports/plots/cohort_retention.png', file_path)
                
                self.logger.info(f"Saved visualizations to: {plots_dir}")
                
            except Exception as e:
                self.logger.warning(f"Could not save visualizations: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Failed to save visualizations: {str(e)}")
    
    def _rename_plot(self, source: str, destination: Path):
        """Rename a plot file"""
        try:
            source_path = Path(source)
            if source_path.exists():
                source_path.rename(destination)
        except Exception:
            pass
    
    def _create_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary DataFrame from results"""
        summary = []
        
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                summary.append({
                    'Section': key,
                    'Type': 'DataFrame',
                    'Shape': f"{value.shape[0]} x {value.shape[1]}",
                    'Size': f"{value.shape[0]} records"
                })
            elif isinstance(value, dict):
                if 'summary' in value:
                    # Extract summary from nested dict
                    for k, v in value['summary'].items():
                        summary.append({
                            'Section': f"{key}.{k}",
                            'Type': type(v).__name__,
                            'Shape': 'value',
                            'Size': str(v)
                        })
                else:
                    summary.append({
                        'Section': key,
                        'Type': 'Dictionary',
                        'Shape': f"{len(value)} keys",
                        'Size': str(list(value.keys()))[:50]
                    })
            elif isinstance(value, (int, float)):
                summary.append({
                    'Section': key,
                    'Type': type(value).__name__,
                    'Shape': 'value',
                    'Size': f"{value:,.2f}"
                })
            else:
                summary.append({
                    'Section': key,
                    'Type': type(value).__name__,
                    'Shape': 'value',
                    'Size': str(value)[:50]
                })
        
        return pd.DataFrame(summary)
    
    def _create_html_content(self, results: Dict[str, Any]) -> str:
        """Create HTML report content"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; background: #ecf0f1; padding: 10px; border-radius: 5px; }
                .section { margin-bottom: 30px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                th { background: #3498db; color: white; padding: 12px; text-align: left; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background: #f5f5f5; }
                .value { background: #2ecc71; color: white; padding: 3px 8px; border-radius: 3px; }
                .footer { margin-top: 40px; text-align: center; color: #7f8c8d; border-top: 1px solid #ddd; padding-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Data Analysis Report</h1>
                <p><strong>Generated:</strong> {current_time}</p>
        """.format(current_time=current_time)
        
        # Add sections
        for key, value in results.items():
            html += f'<div class="section"><h2>📌 {key.replace("_", " ").title()}</h2>'
            
            if isinstance(value, pd.DataFrame):
                html += value.head(10).to_html(index=False)
                if len(value) > 10:
                    html += f'<p>... and {len(value) - 10} more rows</p>'
            
            elif isinstance(value, dict):
                html += '<ul>'
                for k, v in list(value.items())[:10]:
                    if isinstance(v, (int, float)):
                        html += f'<li><strong>{k}:</strong> <span class="value">{v:,.2f}</span></li>'
                    else:
                        html += f'<li><strong>{k}:</strong> {v}</li>'
                html += '</ul>'
            
            else:
                html += f'<p><strong>Value:</strong> {value}</p>'
            
            html += '</div>'
        
        html += """
                <div class="footer">
                    <p>Generated by Data Analysis Pipeline</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html