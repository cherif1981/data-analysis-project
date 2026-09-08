# src/analyzers/forecast_analyzer.py
"""
وحدة التنبؤ بالمبيعات - إصدار لا يعتمد على Prophet
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class ForecastAnalyzer:
    """تحليل التنبؤ بالمبيعات - بدون استخدام Prophet"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.method = 'StatsModels'
        
    def forecast(self, df: pd.DataFrame, periods: int = 12) -> Dict[str, Any]:
        """
        التنبؤ بالمبيعات
        
        Args:
            df: DataFrame يحتوي على بيانات المبيعات
            periods: عدد الفترات للتنبؤ
            
        Returns:
            قاموس يحتوي على نتائج التنبؤ
        """
        try:
            self.logger.info(f"🔮 بدء التنبؤ لـ {periods} فترة...")
            
            # التحقق من البيانات
            if df.empty:
                raise ValueError("البيانات فارغة")
            
            # تجهيز البيانات
            forecast_data = self._prepare_data(df)
            
            # محاولة استخدام أفضل طريقة متاحة
            result = None
            
            # 1. محاولة استخدام ARIMA (statsmodels)
            try:
                self.logger.info("📊 محاولة استخدام ARIMA...")
                result = self._forecast_with_arima(forecast_data, periods)
                if result:
                    self.method = 'ARIMA'
                    self.logger.info("✅ تم استخدام ARIMA بنجاح")
            except Exception as e:
                self.logger.warning(f"⚠️ ARIMA فشل: {str(e)}")
            
            # 2. إذا فشل ARIMA، استخدم المتوسط المتحرك
            if result is None:
                try:
                    self.logger.info("📊 محاولة استخدام المتوسط المتحرك...")
                    result = self._forecast_with_moving_average(forecast_data, periods)
                    if result:
                        self.method = 'Moving Average'
                        self.logger.info("✅ تم استخدام المتوسط المتحرك بنجاح")
                except Exception as e:
                    self.logger.warning(f"⚠️ المتوسط المتحرك فشل: {str(e)}")
            
            # 3. إذا فشل كل شيء، استخدم الطريقة البسيطة
            if result is None:
                self.logger.info("📊 استخدام الطريقة البسيطة...")
                result = self._forecast_simple(forecast_data, periods)
                self.method = 'Simple'
            
            self.logger.info(f"✅ تم إكمال التنبؤ باستخدام طريقة {self.method}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في التنبؤ: {str(e)}")
            raise
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """تجهيز البيانات للتنبؤ"""
        # التأكد من وجود عمود التاريخ
        date_col = 'purchase_date'
        if date_col not in df.columns:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
            else:
                raise ValueError("لا يوجد عمود تاريخ في البيانات")
        
        # تجميع المبيعات حسب التاريخ
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # استخدام عمود المبلغ
        amount_col = 'amount' if 'amount' in df_copy.columns else df_copy.columns[0]
        
        # تجميع يومياً
        daily_sales = df_copy.groupby(df_copy[date_col].dt.date).agg({
            amount_col: 'sum'
        }).reset_index()
        
        daily_sales.columns = ['ds', 'y']
        daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
        
        # التأكد من وجود تواريخ متسلسلة
        date_range = pd.date_range(
            start=daily_sales['ds'].min(), 
            end=daily_sales['ds'].max(), 
            freq='D'
        )
        daily_sales = daily_sales.set_index('ds').reindex(date_range).reset_index()
        daily_sales.columns = ['ds', 'y']
        daily_sales['y'] = daily_sales['y'].fillna(0)
        
        self.logger.info(f"📊 تم تجهيز {len(daily_sales)} يوم للتنبؤ")
        return daily_sales
    
    def _forecast_with_arima(self, data: pd.DataFrame, periods: int) -> Optional[Dict[str, Any]]:
        """التنبؤ باستخدام ARIMA من statsmodels"""
        try:
            # محاولة استيراد statsmodels
            try:
                from statsmodels.tsa.arima.model import ARIMA
                from statsmodels.tsa.stattools import adfuller
            except ImportError:
                self.logger.warning("⚠️ statsmodels غير مثبت")
                return None
            
            # التأكد من وجود بيانات كافية
            if len(data) < 10:
                self.logger.warning("⚠️ البيانات قليلة جداً لـ ARIMA")
                return None
            
            # الحصول على البيانات
            y = data['y'].values
            
            # التحقق من الاستقرارية
            try:
                result = adfuller(y)
                is_stationary = result[1] < 0.05
            except:
                is_stationary = False
            
            if not is_stationary and len(y) > 5:
                # البيانات غير مستقرة، استخدام الفروق
                y_diff = np.diff(y)
                # التأكد من عدم وجود قيم NaN
                y_diff = y_diff[~np.isnan(y_diff)]
                if len(y_diff) > 5:
                    y = y_diff
            
            # تحديد معلمات ARIMA
            try:
                # محاولة العثور على أفضل معلمات
                best_aic = float('inf')
                best_order = (1, 1, 1)
                
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(y, order=(p, d, q))
                                model_fit = model.fit()
                                if model_fit.aic < best_aic:
                                    best_aic = model_fit.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                self.logger.info(f"📊 أفضل معلمات ARIMA: {best_order}")
                
            except:
                # استخدام معلمات افتراضية
                best_order = (1, 1, 1)
            
            # إنشاء نموذج ARIMA
            model = ARIMA(y, order=best_order)
            model_fit = model.fit()
            
            # التنبؤ
            forecast = model_fit.forecast(steps=periods)
            
            # التأكد من عدم وجود قيم سالبة
            forecast = np.maximum(forecast, 0)
            
            # إنشاء التواريخ
            last_date = data['ds'].max()
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
            
            # إنشاء DataFrame للتنبؤ
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast,
                'yhat_lower': forecast * 0.85,
                'yhat_upper': forecast * 1.15
            })
            
            return {
                'model': 'ARIMA',
                'periods': periods,
                'forecast': forecast_df,
                'full_forecast': forecast_df,
                'components': ['trend'],
                'summary': {
                    'mean_forecast': np.mean(forecast),
                    'max_forecast': max(forecast),
                    'min_forecast': min(forecast),
                    'total_forecast': sum(forecast)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في ARIMA: {str(e)}")
            return None
    
    def _forecast_with_moving_average(self, data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """التنبؤ باستخدام المتوسط المتحرك"""
        try:
            y = data['y'].values
            
            # اختيار النافذة المناسبة
            window = min(7, len(y) // 3)
            if window < 2:
                window = 2
            
            # حساب المتوسط المتحرك
            ma = pd.Series(y).rolling(window=window, center=False).mean()
            
            # استخدام آخر قيمة من المتوسط المتحرك
            last_ma = ma.dropna().iloc[-1] if not ma.dropna().empty else np.mean(y[-window:])
            
            # حساب الاتجاه
            if len(y) >= window:
                trend = (y[-1] - y[-window]) / window
            else:
                trend = 0
            
            # حساب التذبذب الموسمي
            seasonal_pattern = []
            if len(y) >= 7:
                # حساب متوسط كل يوم من أيام الأسبوع
                days_of_week = [d.dayofweek for d in data['ds']]
                seasonal_means = {}
                for i in range(7):
                    mask = [d == i for d in days_of_week]
                    if any(mask):
                        seasonal_means[i] = np.mean(y[mask])
                
                if seasonal_means:
                    # آخر يوم في البيانات
                    last_day = data['ds'].iloc[-1].dayofweek
                    for i in range(periods):
                        day_idx = (last_day + i) % 7
                        if day_idx in seasonal_means:
                            seasonal_pattern.append(seasonal_means[day_idx] - np.mean(y))
                        else:
                            seasonal_pattern.append(0)
            
            # إنشاء التنبؤات
            forecast_values = []
            for i in range(periods):
                # إضافة بعض التذبذب العشوائي
                noise = np.random.normal(0, last_ma * 0.05)
                seasonal = seasonal_pattern[i] if i < len(seasonal_pattern) else 0
                next_val = last_ma + trend * (i + 1) + seasonal + noise
                forecast_values.append(max(0, next_val))
            
            # إنشاء التواريخ
            last_date = data['ds'].max()
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
            
            # إنشاء DataFrame
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast_values,
                'yhat_lower': [v * 0.8 for v in forecast_values],
                'yhat_upper': [v * 1.2 for v in forecast_values]
            })
            
            return {
                'model': 'Moving Average with Seasonality',
                'periods': periods,
                'forecast': forecast_df,
                'full_forecast': forecast_df,
                'components': ['trend', 'seasonal'],
                'summary': {
                    'mean_forecast': np.mean(forecast_values),
                    'max_forecast': max(forecast_values),
                    'min_forecast': min(forecast_values),
                    'total_forecast': sum(forecast_values)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في Moving Average: {str(e)}")
            return self._forecast_simple(data, periods)
    
    def _forecast_simple(self, data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """طريقة تنبؤ بسيطة جداً"""
        y = data['y'].values
        
        if len(y) == 0:
            forecast_values = [0] * periods
        else:
            # استخدام المتوسط مع مراعاة الاتجاه
            mean_val = np.mean(y)
            std_val = np.std(y) if len(y) > 1 else mean_val * 0.1
            
            # حساب الاتجاه البسيط
            if len(y) >= 3:
                x = np.arange(len(y))
                z = np.polyfit(x, y, 1)
                trend_slope = z[0]
            else:
                trend_slope = 0
            
            forecast_values = []
            for i in range(periods):
                # إضافة عشوائية بسيطة
                noise = np.random.normal(0, std_val * 0.1)
                next_val = mean_val + trend_slope * (i + 1) + noise
                forecast_values.append(max(0, next_val))
        
        # إنشاء التواريخ
        last_date = data['ds'].max() if not data.empty else pd.Timestamp.now()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values,
            'yhat_lower': [v * 0.85 for v in forecast_values],
            'yhat_upper': [v * 1.15 for v in forecast_values]
        })
        
        return {
            'model': 'Simple Average (Fallback)',
            'periods': periods,
            'forecast': forecast_df,
            'full_forecast': forecast_df,
            'components': ['trend'],
            'summary': {
                'mean_forecast': np.mean(forecast_values) if forecast_values else 0,
                'max_forecast': max(forecast_values) if forecast_values else 0,
                'min_forecast': min(forecast_values) if forecast_values else 0,
                'total_forecast': sum(forecast_values) if forecast_values else 0
            }
        }