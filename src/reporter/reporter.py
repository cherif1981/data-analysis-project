# src/reporter/reporter.py - نسخة معدلة
"""
وحدة إنشاء التقارير
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from typing import Dict, Any, List
import logging
from datetime import datetime

class Reporter:
    """إنشاء التقارير بتنسيقات مختلفة"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_excel_report(self, results: Dict[str, Any], filename: str) -> None:
        """
        إنشاء تقرير Excel متعدد الأوراق
        """
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for sheet_name, data in results.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    elif isinstance(data, dict):
                        # تحويل القاموس إلى DataFrame
                        df = pd.DataFrame([data])
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    elif isinstance(data, (int, float, str)):
                        df = pd.DataFrame({'Value': [data]})
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    else:
                        try:
                            df = pd.DataFrame({'Result': [str(data)]})
                            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                        except:
                            pass
            
            self.logger.info(f"✅ تم إنشاء تقرير Excel: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير Excel: {str(e)}")
            raise
    
    def generate_html_report(self, results: Dict[str, Any], filename: str) -> None:
        """
        إنشاء تقرير HTML تفاعلي
        """
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            html_content = self._create_html_content(results)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ تم إنشاء تقرير HTML: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير HTML: {str(e)}")
            raise
    
    def generate_pdf_report(self, results: Dict[str, Any], filename: str) -> None:
        """
        إنشاء تقرير PDF - معالجة الأخطاء
        """
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # ✅ محاولة إنشاء PDF باستخدام طرق مختلفة
            pdf_created = False
            
            # الطريقة 1: محاولة استخدام weasyprint
            try:
                from weasyprint import HTML
                html_content = self._create_html_content(results)
                HTML(string=html_content).write_pdf(filename)
                self.logger.info(f"✅ تم إنشاء تقرير PDF: {filename}")
                pdf_created = True
            except ImportError:
                self.logger.warning("⚠️ weasyprint غير مثبت")
            except OSError as e:
                self.logger.warning(f"⚠️ خطأ في weasyprint: {str(e)}")
                self.logger.info("📝 سيتم إنشاء HTML بدلاً من PDF")
            
            # الطريقة 2: محاولة استخدام pdfkit
            if not pdf_created:
                try:
                    import pdfkit
                    html_content = self._create_html_content(results)
                    html_filename = filename.replace('.pdf', '_temp.html')
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    pdfkit.from_file(html_filename, filename)
                    os.remove(html_filename)
                    self.logger.info(f"✅ تم إنشاء تقرير PDF: {filename}")
                    pdf_created = True
                except ImportError:
                    self.logger.warning("⚠️ pdfkit غير مثبت")
                except Exception as e:
                    self.logger.warning(f"⚠️ خطأ في pdfkit: {str(e)}")
            
            # الطريقة 3: إنشاء HTML بدلاً من PDF
            if not pdf_created:
                html_filename = filename.replace('.pdf', '.html')
                self.generate_html_report(results, html_filename)
                self.logger.info(f"✅ تم إنشاء تقرير HTML بديل: {html_filename}")
                self.logger.info("💡 نصيحة: قم بتثبيت weasyprint لإنشاء PDF: pip install weasyprint")
                
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير PDF: {str(e)}")
            # إنشاء HTML كحل بديل
            try:
                html_filename = filename.replace('.pdf', '.html')
                self.generate_html_report(results, html_filename)
                self.logger.info(f"✅ تم إنشاء تقرير HTML بديل: {html_filename}")
            except:
                pass
    
    def generate_json_report(self, results: Dict[str, Any], filename: str) -> None:
        """
        إنشاء تقرير JSON
        """
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # تحويل DataFrame إلى JSON
            json_data = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    json_data[key] = value.to_dict(orient='records')
                elif isinstance(value, dict):
                    json_data[key] = value
                else:
                    json_data[key] = str(value)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ تم إنشاء تقرير JSON: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير JSON: {str(e)}")
            raise
    
    def _create_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """إنشاء جدول ملخص للنتائج"""
        summary = []
        
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                summary.append({
                    'Metric': key,
                    'Type': 'DataFrame',
                    'Shape': f"{value.shape[0]} rows, {value.shape[1]} columns",
                    'Value': f"{value.shape[0]} records"
                })
            elif isinstance(value, dict):
                summary.append({
                    'Metric': key,
                    'Type': 'Dictionary',
                    'Shape': f"{len(value)} keys",
                    'Value': str(list(value.keys()))[:50] + '...'
                })
            elif isinstance(value, (int, float)):
                summary.append({
                    'Metric': key,
                    'Type': type(value).__name__,
                    'Shape': '1 value',
                    'Value': f"{value:,.2f}"
                })
            else:
                summary.append({
                    'Metric': key,
                    'Type': type(value).__name__,
                    'Shape': 'N/A',
                    'Value': str(value)[:50]
                })
        
        return pd.DataFrame(summary)
    
    def _create_html_content(self, results: Dict[str, Any]) -> str:
        """إنشاء محتوى HTML للتقرير"""
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        num_sections = len(results)
        
        html_parts = []
        
        # بداية HTML
        html_parts.append('''
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>تقرير تحليل البيانات</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #34495e;
                    margin-top: 30px;
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-radius: 5px;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }
                th {
                    background-color: #3498db;
                    color: white;
                    padding: 12px;
                    text-align: right;
                }
                td {
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .value-highlight {
                    background-color: #2ecc71;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                .summary-box {
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    border-right: 4px solid #3498db;
                }
                .footer {
                    margin-top: 40px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 12px;
                    border-top: 1px solid #ddd;
                    padding-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 تقرير تحليل البيانات</h1>
                <p>تاريخ التقرير: ''' + current_time + '''</p>
        ''')
        
        # إضافة الأقسام المختلفة
        for section_name, section_data in results.items():
            html_parts.append('<div class="section">')
            html_parts.append('<h2>📌 ' + section_name.replace('_', ' ').title() + '</h2>')
            
            if isinstance(section_data, pd.DataFrame):
                html_parts.append(section_data.head(10).to_html(index=False, classes='table'))
                if len(section_data) > 10:
                    html_parts.append('<p style="color: #7f8c8d;">... وعرض ' + str(len(section_data) - 10) + ' صفوف إضافية</p>')
            
            elif isinstance(section_data, dict):
                html_parts.append('<ul>')
                for k, v in list(section_data.items())[:10]:
                    if isinstance(v, (int, float)):
                        html_parts.append('<li><strong>' + str(k) + ':</strong> <span class="value-highlight">' + f"{v:,.2f}" + '</span></li>')
                    else:
                        html_parts.append('<li><strong>' + str(k) + ':</strong> ' + str(v) + '</li>')
                html_parts.append('</ul>')
            
            else:
                html_parts.append('<p><strong>القيمة:</strong> ' + str(section_data) + '</p>')
            
            html_parts.append('</div>')
        
        # إضافة ملخص ونهاية HTML
        html_parts.append('''
                <div class="summary-box">
                    <h3>📋 ملخص التحليل</h3>
                    <ul>
                        <li>✅ تم تحليل البيانات بنجاح</li>
                        <li>📊 تم إنشاء ''' + str(num_sections) + ''' قسم تحليل</li>
                        <li>📁 تم حفظ التقارير في مجلد reports</li>
                    </ul>
                </div>
                <div class="footer">
                    <p>تم إنشاء هذا التقرير بواسطة نظام تحليل البيانات التلقائي</p>
                    <p>جميع الحقوق محفوظة © 2026</p>
                </div>
            </div>
        </body>
        </html>
        ''')
        
        return ''.join(html_parts)