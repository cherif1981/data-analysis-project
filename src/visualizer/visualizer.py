# src/visualizer/visualizer.py
"""
وحدة التصور والرسوم البيانية
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional

class Visualizer:
    """إنشاء التصورات والرسوم البيانية"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # ✅ إعداد نمط الرسوم - استخدام أنماط متوافقة
        self._setup_style()
        
    def _setup_style(self):
        """إعداد نمط الرسوم البيانية"""
        try:
            # محاولة استخدام أنماط seaborn المتوفرة
            available_styles = plt.style.available
            
            # قائمة بأنماط seaborn المحتملة (حسب الإصدار)
            seaborn_styles = [
                'seaborn-v0_8-darkgrid',
                'seaborn-darkgrid',
                'seaborn-v0_8-dark',
                'seaborn-dark',
                'seaborn-v0_8-whitegrid',
                'seaborn-whitegrid',
                'seaborn-v0_8-talk',
                'seaborn-talk'
            ]
            
            # اختيار أول نمط متوفر
            selected_style = None
            for style in seaborn_styles:
                if style in available_styles:
                    selected_style = style
                    break
            
            if selected_style:
                plt.style.use(selected_style)
                self.logger.info(f"✅ تم استخدام نمط: {selected_style}")
            else:
                # استخدام نمط default إذا لم يتوفر seaborn
                plt.style.use('default')
                self.logger.info("✅ تم استخدام النمط الافتراضي")
                
            # تعيين لوحة ألوان seaborn
            try:
                sns.set_palette("husl")
            except:
                sns.set_palette("deep")
                
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في إعداد النمط: {str(e)}")
            # استخدام النمط الافتراضي
            plt.style.use('default')
    
    def plot_rfm_distribution(self, rfm_data: pd.DataFrame) -> None:
        """رسم توزيع RFM"""
        try:
            if rfm_data is None or rfm_data.empty:
                self.logger.warning("⚠️ لا توجد بيانات RFM للرسم")
                return
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # توزيع Recency
            if 'recency' in rfm_data.columns:
                axes[0].hist(rfm_data['recency'], bins=30, color='blue', alpha=0.7, edgecolor='black')
                axes[0].set_title('توزيع Recency')
                axes[0].set_xlabel('عدد الأيام')
                axes[0].set_ylabel('عدد العملاء')
                axes[0].grid(True, alpha=0.3)
            
            # توزيع Frequency
            if 'frequency' in rfm_data.columns:
                axes[1].hist(rfm_data['frequency'], bins=30, color='green', alpha=0.7, edgecolor='black')
                axes[1].set_title('توزيع Frequency')
                axes[1].set_xlabel('عدد الطلبات')
                axes[1].set_ylabel('عدد العملاء')
                axes[1].grid(True, alpha=0.3)
            
            # توزيع Monetary
            if 'monetary' in rfm_data.columns:
                axes[2].hist(rfm_data['monetary'], bins=30, color='red', alpha=0.7, edgecolor='black')
                axes[2].set_title('توزيع Monetary')
                axes[2].set_xlabel('قيمة المشتريات')
                axes[2].set_ylabel('عدد العملاء')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # التأكد من وجود المجلد
            Path('reports/plots').mkdir(parents=True, exist_ok=True)
            plt.savefig('reports/plots/rfm_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info("✅ تم إنشاء رسم توزيع RFM")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في رسم RFM: {str(e)}")
    
    def plot_forecast(self, forecast_data: Dict[str, Any]) -> None:
        """رسم التنبؤ"""
        try:
            if not forecast_data or 'forecast' not in forecast_data:
                self.logger.warning("⚠️ لا توجد بيانات تنبؤ للرسم")
                return
            
            forecast_df = forecast_data['forecast']
            
            if forecast_df is None or forecast_df.empty:
                self.logger.warning("⚠️ بيانات التنبؤ فارغة")
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # رسم البيانات الفعلية والتنبؤية
            if 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
                ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                       label='التنبؤ', color='blue', linewidth=2)
                
                # إضافة فترات الثقة
                if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    ax.fill_between(forecast_df['ds'], 
                                   forecast_df['yhat_lower'], 
                                   forecast_df['yhat_upper'],
                                   alpha=0.2, color='blue', label='فترة الثقة')
                
                ax.set_title('التنبؤ بالمبيعات')
                ax.set_xlabel('التاريخ')
                ax.set_ylabel('المبيعات')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                Path('reports/plots').mkdir(parents=True, exist_ok=True)
                plt.savefig('reports/plots/forecast.png', dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info("✅ تم إنشاء رسم التنبؤ")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في رسم التنبؤ: {str(e)}")
    
    def plot_seasonal_decomposition(self, seasonal_result: Dict[str, Any]) -> None:
        """رسم التحليل الموسمي"""
        try:
            if not seasonal_result:
                self.logger.warning("⚠️ لا توجد بيانات موسمية للرسم")
                return
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # رسم المكونات إذا كانت موجودة
            if 'observed' in seasonal_result and seasonal_result['observed'] is not None:
                seasonal_result['observed'].plot(ax=axes[0], title='البيانات الأصلية', color='blue')
                axes[0].grid(True, alpha=0.3)
            
            if 'trend' in seasonal_result and seasonal_result['trend'] is not None:
                seasonal_result['trend'].plot(ax=axes[1], title='الاتجاه', color='green')
                axes[1].grid(True, alpha=0.3)
            
            if 'seasonal' in seasonal_result and seasonal_result['seasonal'] is not None:
                seasonal_result['seasonal'].plot(ax=axes[2], title='المكون الموسمي', color='orange')
                axes[2].grid(True, alpha=0.3)
            
            if 'resid' in seasonal_result and seasonal_result['resid'] is not None:
                seasonal_result['resid'].plot(ax=axes[3], title='المتبقي', color='red')
                axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            Path('reports/plots').mkdir(parents=True, exist_ok=True)
            plt.savefig('reports/plots/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info("✅ تم إنشاء رسم التحليل الموسمي")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في رسم التحليل الموسمي: {str(e)}")
    
    def plot_cohort_retention(self, retention_result: Dict[str, Any]) -> None:
        """رسم مصفوفة الاحتفاظ بالعملاء"""
        try:
            if not retention_result or 'retention_rates' not in retention_result:
                self.logger.warning("⚠️ لا توجد بيانات احتفاظ للرسم")
                return
            
            retention_rates = retention_result['retention_rates']
            
            if retention_rates is None or retention_rates.empty:
                self.logger.warning("⚠️ مصفوفة الاحتفاظ فارغة")
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # رسم الخريطة الحرارية
            sns.heatmap(
                retention_rates, 
                annot=True, 
                fmt='.0f', 
                cmap='YlGnBu', 
                ax=ax,
                cbar_kws={'label': 'نسبة الاحتفاظ (%)'}
            )
            
            ax.set_title('مصفوفة الاحتفاظ بالعملاء (Cohort Analysis)')
            ax.set_xlabel('الفترة (شهر)')
            ax.set_ylabel('المجموعة الزمنية (شهر البدء)')
            
            plt.tight_layout()
            
            Path('reports/plots').mkdir(parents=True, exist_ok=True)
            plt.savefig('reports/plots/cohort_retention.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("✅ تم حفظ رسم مصفوفة الاحتفاظ")
            
            # رسم منحنى الاحتفاظ
            self._plot_retention_curve(retention_result)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في رسم الاحتفاظ: {str(e)}")
    
    def _plot_retention_curve(self, retention_result: Dict[str, Any]) -> None:
        """رسم منحنى الاحتفاظ"""
        try:
            retention_rates = retention_result.get('retention_rates')
            
            if retention_rates is None or retention_rates.empty:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # حساب متوسط الاحتفاظ لكل فترة
            avg_retention = retention_rates.mean()
            
            # رسم المنحنى
            ax.plot(avg_retention.index, avg_retention.values, 
                    marker='o', linewidth=2, markersize=8, color='blue')
            
            ax.set_title('متوسط الاحتفاظ بالعملاء عبر الفترات')
            ax.set_xlabel('الفترة (شهر)')
            ax.set_ylabel('نسبة الاحتفاظ (%)')
            ax.grid(True, alpha=0.3)
            
            # إضافة خطوط إرشادية
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
            ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='25%')
            
            # إضافة القيم على النقاط
            for i, v in enumerate(avg_retention.values):
                if not np.isnan(v):
                    ax.annotate(f'{v:.0f}%', 
                               (avg_retention.index[i], v),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center',
                               fontsize=9)
            
            ax.legend()
            plt.tight_layout()
            
            Path('reports/plots').mkdir(parents=True, exist_ok=True)
            plt.savefig('reports/plots/retention_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("✅ تم حفظ منحنى الاحتفاظ")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في رسم منحنى الاحتفاظ: {str(e)}")
    
    def save_plots(self, output_dir: str = 'reports/plots/') -> None:
        """حفظ جميع الرسوم البيانية"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 تم حفظ الرسوم في: {output_dir}")
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ الرسوم: {str(e)}")