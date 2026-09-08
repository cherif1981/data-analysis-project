# src/visualizer/dashboard.py
"""
Interactive Dashboard Generator
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime

class DashboardGenerator:
    """
    Generate interactive dashboard with visualizations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.output_dir = self.config.get('output', {}).get('plots_dir', 'outputs/plots')
        self.reports_dir = self.config.get('output', {}).get('reports_dir', 'outputs/reports')
        
        # Ensure directories exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
    
    def generate(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate complete dashboard
        
        Args:
            results: Dictionary containing analysis results
            
        Returns:
            Dictionary with generated file paths
        """
        try:
            self.logger.info("Generating dashboard...")
            
            generated_files = {}
            
            # Generate individual plots
            if 'rfm' in results:
                self._plot_rfm_dashboard(results['rfm'])
                generated_files['rfm'] = f"{self.output_dir}/rfm_dashboard.png"
            
            if 'kpis' in results:
                self._plot_kpi_dashboard(results['kpis'])
                generated_files['kpis'] = f"{self.output_dir}/kpi_dashboard.png"
            
            if 'forecast' in results:
                self._plot_forecast_dashboard(results['forecast'])
                generated_files['forecast'] = f"{self.output_dir}/forecast_dashboard.png"
            
            if 'retention' in results:
                self._plot_retention_dashboard(results['retention'])
                generated_files['retention'] = f"{self.output_dir}/retention_dashboard.png"
            
            if 'churn' in results:
                self._plot_churn_dashboard(results['churn'])
                generated_files['churn'] = f"{self.output_dir}/churn_dashboard.png"
            
            # Generate summary dashboard
            self._plot_summary_dashboard(results)
            generated_files['summary'] = f"{self.output_dir}/summary_dashboard.png"
            
            # Generate HTML dashboard
            html_file = self._generate_html_dashboard(results)
            generated_files['html'] = html_file
            
            self.logger.info(f"Dashboard generated: {len(generated_files)} files")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {str(e)}")
            return {}
    
    def _plot_rfm_dashboard(self, rfm_data: Dict[str, Any]):
        """Create RFM dashboard"""
        try:
            if 'data' not in rfm_data:
                return
            
            df = rfm_data['data']
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. RFM Distribution
            ax = axes[0, 0]
            if 'segment' in df.columns:
                segment_counts = df['segment'].value_counts()
                colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
                ax.pie(segment_counts.values, labels=segment_counts.index, 
                       autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Customer Segments Distribution')
            
            # 2. RFM Scores Heatmap
            ax = axes[0, 1]
            if all(col in df.columns for col in ['recency', 'frequency', 'monetary']):
                # Create RFM score matrix
                rfm_matrix = df.groupby('segment')[['recency', 'frequency', 'monetary']].mean()
                sns.heatmap(rfm_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
                ax.set_title('Average RFM by Segment')
            
            # 3. Monetary Distribution by Segment
            ax = axes[1, 0]
            if 'segment' in df.columns and 'monetary' in df.columns:
                df.boxplot(column='monetary', by='segment', ax=ax)
                ax.set_title('Monetary Distribution by Segment')
                ax.set_xlabel('Segment')
                ax.set_ylabel('Monetary Value')
            
            # 4. Customer Count by Segment
            ax = axes[1, 1]
            if 'segment' in df.columns:
                counts = df['segment'].value_counts()
                bars = ax.bar(counts.index, counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(counts))))
                ax.set_title('Customer Count by Segment')
                ax.set_xlabel('Segment')
                ax.set_ylabel('Number of Customers')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(value), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/rfm_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("RFM dashboard created")
            
        except Exception as e:
            self.logger.error(f"RFM dashboard failed: {str(e)}")
    
    def _plot_kpi_dashboard(self, kpis: Dict[str, Any]):
        """Create KPI dashboard"""
        try:
            if not kpis:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. KPI Overview
            ax = axes[0, 0]
            kpi_items = {k: v for k, v in kpis.items() if isinstance(v, (int, float))}
            if kpi_items:
                names = list(kpi_items.keys())[:8]
                values = list(kpi_items.values())[:8]
                colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
                bars = ax.barh(names, values, color=colors)
                ax.set_title('Key Performance Indicators')
                ax.set_xlabel('Value')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_width() + max(values) * 0.01, 
                           bar.get_y() + bar.get_height()/2,
                           f'{value:,.0f}', ha='left', va='center')
            
            # 2. Revenue Trend (if available)
            ax = axes[0, 1]
            ax.text(0.5, 0.5, 'Revenue Trends\n(Add time series data)', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Revenue Trend')
            
            # 3. Customer Metrics
            ax = axes[1, 0]
            customer_metrics = {k: v for k, v in kpis.items() if 'customer' in k.lower() or 'retention' in k.lower()}
            if customer_metrics:
                names = list(customer_metrics.keys())
                values = list(customer_metrics.values())
                bars = ax.bar(names, values, color=plt.cm.Greens(np.linspace(0.3, 0.9, len(names))))
                ax.set_title('Customer Metrics')
                ax.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:,.0f}', ha='center', va='bottom')
            
            # 4. Summary
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = "📊 KPI Summary\n\n"
            for k, v in list(kpi_items.items())[:10]:
                if isinstance(v, (int, float)):
                    summary_text += f"• {k}: {v:,.2f}\n"
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/kpi_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("KPI dashboard created")
            
        except Exception as e:
            self.logger.error(f"KPI dashboard failed: {str(e)}")
    
    def _plot_forecast_dashboard(self, forecast: Dict[str, Any]):
        """Create forecast dashboard"""
        try:
            if 'forecast' not in forecast:
                return
            
            forecast_df = forecast['forecast']
            if forecast_df.empty:
                return
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # 1. Forecast Plot
            ax = axes[0]
            if 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
                ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                       label='Forecast', color='blue', linewidth=2)
                
                if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    ax.fill_between(forecast_df['ds'], 
                                   forecast_df['yhat_lower'], 
                                   forecast_df['yhat_upper'],
                                   alpha=0.2, color='blue', label='Confidence Interval')
                
                ax.set_title('Sales Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            
            # 2. Forecast Summary
            ax = axes[1]
            ax.axis('off')
            
            summary_text = "📈 Forecast Summary\n\n"
            if 'summary' in forecast:
                summary = forecast['summary']
                summary_text += f"Mean Forecast: ${summary.get('mean_forecast', 0):,.2f}\n"
                summary_text += f"Max Forecast: ${summary.get('max_forecast', 0):,.2f}\n"
                summary_text += f"Min Forecast: ${summary.get('min_forecast', 0):,.2f}\n"
                summary_text += f"Total Forecast: ${summary.get('total_forecast', 0):,.2f}\n"
            
            if 'model' in forecast:
                summary_text += f"\nModel: {forecast['model']}\n"
            if 'periods' in forecast:
                summary_text += f"Periods: {forecast['periods']}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/forecast_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Forecast dashboard created")
            
        except Exception as e:
            self.logger.error(f"Forecast dashboard failed: {str(e)}")
    
    def _plot_retention_dashboard(self, retention: Dict[str, Any]):
        """Create retention dashboard"""
        try:
            if 'retention_rates' not in retention:
                return
            
            retention_rates = retention['retention_rates']
            if retention_rates.empty:
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. Retention Matrix Heatmap
            ax = axes[0]
            sns.heatmap(retention_rates, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
            ax.set_title('Retention Matrix')
            ax.set_xlabel('Period')
            ax.set_ylabel('Cohort')
            
            # 2. Average Retention Curve
            ax = axes[1]
            avg_retention = retention_rates.mean()
            ax.plot(avg_retention.index, avg_retention.values, 
                   marker='o', linewidth=2, markersize=8, color='green')
            ax.set_title('Average Retention Rate')
            ax.set_xlabel('Period')
            ax.set_ylabel('Retention Rate (%)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(avg_retention.values):
                if not np.isnan(v):
                    ax.annotate(f'{v:.0f}%', 
                               (avg_retention.index[i], v),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/retention_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Retention dashboard created")
            
        except Exception as e:
            self.logger.error(f"Retention dashboard failed: {str(e)}")
    
    def _plot_churn_dashboard(self, churn: Dict[str, Any]):
        """Create churn dashboard"""
        try:
            if 'predictions' not in churn:
                return
            
            predictions = churn['predictions']
            if not predictions:
                return
            
            # Create DataFrame from predictions
            df = pd.DataFrame(predictions)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Risk Distribution
            ax = axes[0, 0]
            if 'churn_risk' in df.columns:
                risk_counts = df['churn_risk'].value_counts()
                colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 
                         'LOW': 'lightgreen', 'VERY LOW': 'green'}
                bars = ax.bar(risk_counts.index, risk_counts.values, 
                             color=[colors.get(r, 'gray') for r in risk_counts.index])
                ax.set_title('Churn Risk Distribution')
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Number of Customers')
                
                for bar, value in zip(bars, risk_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(value), ha='center', va='bottom')
            
            # 2. Churn Probability Distribution
            ax = axes[0, 1]
            if 'churn_probability' in df.columns:
                ax.hist(df['churn_probability'], bins=20, color='blue', alpha=0.7, edgecolor='black')
                ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
                ax.axvline(x=0.7, color='orange', linestyle='--', label='High Risk (0.7)')
                ax.set_title('Churn Probability Distribution')
                ax.set_xlabel('Probability')
                ax.set_ylabel('Count')
                ax.legend()
            
            # 3. Top 10 High Risk Customers
            ax = axes[1, 0]
            if 'churn_probability' in df.columns and 'customer_id' in df.columns:
                top_risk = df.nlargest(10, 'churn_probability')
                bars = ax.barh(top_risk['customer_id'].astype(str), 
                              top_risk['churn_probability'],
                              color=plt.cm.Reds(np.linspace(0.3, 0.9, len(top_risk))))
                ax.set_title('Top 10 High Risk Customers')
                ax.set_xlabel('Churn Probability')
                
                for bar, prob in zip(bars, top_risk['churn_probability']):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{prob:.1%}', ha='left', va='center')
            
            # 4. Summary Statistics
            ax = axes[1, 1]
            ax.axis('off')
            
            summary = churn.get('summary', {})
            summary_text = "🔴 Churn Risk Summary\n\n"
            summary_text += f"Total Customers: {summary.get('total_customers', 0)}\n"
            summary_text += f"Average Risk: {summary.get('avg_churn_probability', 0):.1%}\n"
            summary_text += f"High Risk: {summary.get('high_risk_count', 0)}\n"
            summary_text += f"Critical Risk: {summary.get('critical_risk_count', 0)}\n\n"
            
            risk_dist = summary.get('risk_distribution', {})
            summary_text += "Risk Distribution:\n"
            for risk, count in risk_dist.items():
                summary_text += f"  • {risk}: {count}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/churn_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Churn dashboard created")
            
        except Exception as e:
            self.logger.error(f"Churn dashboard failed: {str(e)}")
    
    def _plot_summary_dashboard(self, results: Dict[str, Any]):
        """Create summary dashboard with all key metrics"""
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle('📊 Data Analysis Summary Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Key Metrics
            ax = fig.add_subplot(gs[0, 0])
            ax.axis('off')
            metrics_text = "📈 Key Metrics\n"
            
            if 'kpis' in results:
                kpis = results['kpis']
                for k, v in list(kpis.items())[:6]:
                    if isinstance(v, (int, float)):
                        metrics_text += f"• {k}: {v:,.2f}\n"
            
            ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
            
            # 2. Quality Score
            ax = fig.add_subplot(gs[0, 1])
            if 'quality_report' in results:
                quality = results['quality_report']
                score = quality.get('quality_score', 0)
                
                # Gauge chart
                colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
                color_idx = min(int(score / 20), 4)
                
                ax.pie([score, 100-score], startangle=90, 
                       colors=[colors[color_idx], 'lightgray'],
                       wedgeprops={'width': 0.3})
                ax.text(0, 0, f'{score:.1f}%', ha='center', va='center', fontsize=20, fontweight='bold')
                ax.set_title('Data Quality Score', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No Quality Data', ha='center', va='center')
            
            # 3. Customer Segments
            ax = fig.add_subplot(gs[0, 2])
            if 'rfm' in results and 'segment_counts' in results['rfm']:
                segments = results['rfm']['segment_counts']
                if segments:
                    # Pie chart
                    ax.pie(segments.values(), labels=segments.keys(), autopct='%1.1f%%')
                    ax.set_title('Customer Segments', fontsize=12)
            
            # 4. Forecast
            ax = fig.add_subplot(gs[1, :])
            if 'forecast' in results and 'forecast' in results['forecast']:
                forecast_df = results['forecast']['forecast']
                if not forecast_df.empty and 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
                    ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                           color='blue', linewidth=2, label='Forecast')
                    ax.fill_between(forecast_df['ds'], 
                                   forecast_df.get('yhat_lower', forecast_df['yhat'] * 0.9), 
                                   forecast_df.get('yhat_upper', forecast_df['yhat'] * 1.1),
                                   alpha=0.2, color='blue')
                    ax.set_title('Sales Forecast', fontsize=12)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
            
            # 5. Retention
            ax = fig.add_subplot(gs[2, 0:2])
            if 'retention' in results and 'retention_rates' in results['retention']:
                retention_rates = results['retention']['retention_rates']
                if not retention_rates.empty:
                    avg_retention = retention_rates.mean()
                    ax.plot(avg_retention.index, avg_retention.values, 
                           marker='o', linewidth=2, markersize=8, color='green')
                    ax.set_title('Average Retention Rate', fontsize=12)
                    ax.set_xlabel('Period')
                    ax.set_ylabel('Retention (%)')
                    ax.grid(True, alpha=0.3)
            
            # 6. Churn
            ax = fig.add_subplot(gs[2, 2])
            if 'churn' in results and 'summary' in results['churn']:
                summary = results['churn']['summary']
                risk_dist = summary.get('risk_distribution', {})
                if risk_dist:
                    colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow',
                             'low': 'lightgreen', 'very_low': 'green'}
                    ax.pie(risk_dist.values(), labels=risk_dist.keys(), 
                           colors=[colors.get(k.lower(), 'gray') for k in risk_dist.keys()],
                           autopct='%1.1f%%')
                    ax.set_title('Churn Risk Distribution', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/summary_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Summary dashboard created")
            
        except Exception as e:
            self.logger.error(f"Summary dashboard failed: {str(e)}")
    
    def _generate_html_dashboard(self, results: Dict[str, Any]) -> str:
        """Generate HTML dashboard with interactive elements"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Data Analysis Dashboard</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
                    .container {{ max-width: 1400px; margin: 0 auto; }}
                    .header {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    .header h1 {{ color: #2c3e50; }}
                    .header p {{ color: #7f8c8d; margin-top: 5px; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                    .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    .card h3 {{ color: #34495e; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    .metric {{ display: inline-block; margin: 10px 20px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                    .metric-label {{ font-size: 12px; color: #7f8c8d; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                    th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
                    td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                    tr:hover {{ background: #f5f5f5; }}
                    .badge {{ padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
                    .badge-critical {{ background: #e74c3c; color: white; }}
                    .badge-high {{ background: #e67e22; color: white; }}
                    .badge-medium {{ background: #f1c40f; color: black; }}
                    .badge-low {{ background: #2ecc71; color: white; }}
                    .badge-very-low {{ background: #27ae60; color: white; }}
                    .footer {{ text-align: center; margin-top: 20px; color: #7f8c8d; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Data Analysis Dashboard</h1>
                        <p>Generated: {timestamp}</p>
                    </div>
                    
                    <div class="grid">
                        <!-- KPIs -->
                        <div class="card">
                            <h3>📈 Key Performance Indicators</h3>
                            <div style="margin: 10px 0;">
            """
            
            # Add KPIs
            if 'kpis' in results:
                kpis = results['kpis']
                for k, v in list(kpis.items())[:8]:
                    if isinstance(v, (int, float)):
                        html_content += f"""
                                <div class="metric">
                                    <div class="metric-value">{v:,.2f}</div>
                                    <div class="metric-label">{k.replace('_', ' ').title()}</div>
                                </div>
                        """
            
            html_content += """
                            </div>
                        </div>
                        
                        <!-- Quality Report -->
                        <div class="card">
                            <h3>📊 Data Quality</h3>
            """
            
            if 'quality_report' in results:
                quality = results['quality_report']
                html_content += f"""
                            <div class="metric">
                                <div class="metric-value">{quality.get('quality_score', 0):.1f}%</div>
                                <div class="metric-label">Quality Score</div>
                            </div>
                            <div style="margin-top: 10px;">
                                <p>Duplicates: {quality.get('duplicates', {}).get('count', 0)}</p>
                                <p>Missing Values: {quality.get('missing_values', {}).get('total', 0)}</p>
                                <p>Invalid Dates: {quality.get('invalid_dates', {}).get('count', 0)}</p>
                            </div>
                """
            
            html_content += """
                        </div>
                    </div>
                    
                    <div class="grid">
                        <!-- RFM Segments -->
                        <div class="card">
                            <h3>👥 Customer Segments</h3>
                            <table>
                                <thead>
                                    <tr><th>Segment</th><th>Count</th><th>%</th></tr>
                                </thead>
                                <tbody>
            """
            
            if 'rfm' in results and 'segment_counts' in results['rfm']:
                total = sum(results['rfm']['segment_counts'].values())
                for segment, count in results['rfm']['segment_counts'].items():
                    percentage = (count / total * 100) if total > 0 else 0
                    html_content += f"""
                                    <tr>
                                        <td>{segment}</td>
                                        <td>{count}</td>
                                        <td>{percentage:.1f}%</td>
                                    </tr>
                    """
            
            html_content += """
                                </tbody>
                            </table>
                        </div>
                        
                        <!-- High Risk Customers -->
                        <div class="card">
                            <h3>🔴 High Risk Customers</h3>
                            <table>
                                <thead>
                                    <tr><th>Customer</th><th>Risk Level</th><th>Probability</th></tr>
                                </thead>
                                <tbody>
            """
            
            if 'churn' in results and 'top_10_risk' in results['churn']:
                for customer in results['churn']['top_10_risk'][:5]:
                    risk_class = customer.get('churn_risk', '').lower().replace(' ', '-')
                    html_content += f"""
                                    <tr>
                                        <td>{customer.get('customer_id', 'N/A')}</td>
                                        <td><span class="badge badge-{risk_class}">{customer.get('churn_risk', 'Unknown')}</span></td>
                                        <td>{customer.get('churn_probability', 0)*100:.1f}%</td>
                                    </tr>
                    """
            
            html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="grid">
                        <!-- Forecast -->
                        <div class="card">
                            <h3>🔮 Forecast Summary</h3>
            """
            
            if 'forecast' in results and 'summary' in results['forecast']:
                summary = results['forecast']['summary']
                html_content += f"""
                            <div style="margin: 10px 0;">
                                <div class="metric">
                                    <div class="metric-value">${summary.get('mean_forecast', 0):,.2f}</div>
                                    <div class="metric-label">Mean Forecast</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">${summary.get('total_forecast', 0):,.2f}</div>
                                    <div class="metric-label">Total Forecast</div>
                                </div>
                            </div>
                            <p>Model: {results['forecast'].get('model', 'Unknown')}</p>
                            <p>Periods: {results['forecast'].get('periods', 0)}</p>
                """
            
            html_content += """
                        </div>
                        
                        <!-- Retention -->
                        <div class="card">
                            <h3>🔄 Retention</h3>
            """
            
            if 'retention' in results and 'summary' in results['retention']:
                retention_summary = results['retention']['summary']
                avg_retention = retention_summary.get('avg_retention', {})
                html_content += f"""
                            <div style="margin: 10px 0;">
                                <div class="metric">
                                    <div class="metric-value">{retention_summary.get('total_customers', 0)}</div>
                                    <div class="metric-label">Total Customers</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">{retention_summary.get('num_cohorts', 0)}</div>
                                    <div class="metric-label">Cohorts</div>
                                </div>
                            </div>
                """
                if avg_retention:
                    html_content += "<p>Average Retention by Period:</p><ul>"
                    for period, rate in list(avg_retention.items())[:5]:
                        html_content += f"<li>Month {period}: {rate:.1f}%</li>"
                    html_content += "</ul>"
            
            html_content += """
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>Generated by Data Analysis Pipeline | All rights reserved © 2026</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            html_file = f"{self.reports_dir}/dashboard.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML dashboard saved: {html_file}")
            return html_file
            
        except Exception as e:
            self.logger.error(f"HTML dashboard failed: {str(e)}")
            return ""