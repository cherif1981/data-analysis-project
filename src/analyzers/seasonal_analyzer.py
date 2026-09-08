# src/analyzers/seasonal_analyzer.py
"""
التحليل الموسمي للبيانات
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# محاولة استيراد statsmodels
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    seasonal_decompose = None
    adfuller = None
    print("⚠️ statsmodels غير مثبت. سيتم استخدام طرق تحليل موسمي بديلة.")

class SeasonalAnalyzer:
    """تحليل المكونات الموسمية للبيانات"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("⚠️ statsmodels غير مثبت، سيتم استخدام تحليل موسمي مبسط")
    
    def decompose(self, df: pd.DataFrame, period: Optional[int] = None) -> Dict[str, Any]:
        """
        تحليل المكونات الموسمية
        
        Args:
            df: DataFrame يحتوي على البيانات
            period: الفترة الموسمية (افتراضي: 7 أيام)
            
        Returns:
            قاموس يحتوي على مكونات التحليل الموسمي
        """
        try:
            self.logger.info("📅 بدء التحليل الموسمي...")
            
            # التحقق من البيانات
            if df.empty:
                raise ValueError("البيانات فارغة")
            
            # تجهيز البيانات
            ts_data = self._prepare_time_series(df)
            
            # اختيار الفترة الموسمية
            if period is None:
                period = self._detect_seasonality(ts_data)
            
            self.logger.info(f"📊 الفترة الموسمية المقترحة: {period} يوم")
            
            # محاولة استخدام statsmodels
            if STATSMODELS_AVAILABLE and len(ts_data) > period * 2:
                try:
                    result = self._decompose_with_statsmodels(ts_data, period)
                    if result:
                        return result
                except Exception as e:
                    self.logger.warning(f"⚠️ فشل تحليل statsmodels: {str(e)}")
            
            # استخدام الطريقة البديلة
            self.logger.info("📊 استخدام الطريقة البديلة للتحليل الموسمي...")
            return self._decompose_alternative(ts_data, period)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في التحليل الموسمي: {str(e)}")
            # إرجاع نتائج بسيطة
            return self._simple_result(e)
    
    def _prepare_time_series(self, df: pd.DataFrame) -> pd.Series:
        """تجهيز البيانات كسلسلة زمنية"""
        # تحديد عمود التاريخ
        date_col = 'purchase_date'
        if date_col not in df.columns:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
            else:
                raise ValueError("لا يوجد عمود تاريخ في البيانات")
        
        # تحديد عمود المبلغ
        amount_col = 'amount' if 'amount' in df.columns else df.columns[0]
        
        # نسخ البيانات
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # تجميع يومياً
        daily_data = df_copy.groupby(df_copy[date_col].dt.date).agg({
            amount_col: 'sum'
        }).reset_index()
        
        daily_data.columns = ['date', 'value']
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        
        # إنشاء سلسلة زمنية متسلسلة
        date_range = pd.date_range(
            start=daily_data['date'].min(),
            end=daily_data['date'].max(),
            freq='D'
        )
        
        ts = daily_data.set_index('date').reindex(date_range).fillna(0)
        ts = ts['value']
        
        self.logger.info(f"📊 تم تجهيز {len(ts)} يوم للتحليل الموسمي")
        return ts
    
    def _detect_seasonality(self, ts: pd.Series) -> int:
        """اكتشاف الفترة الموسمية تلقائياً"""
        try:
            # الفترات الموسمية الشائعة
            if len(ts) >= 365:
                return 365  # سنوي
            elif len(ts) >= 90:
                return 30   # شهري
            elif len(ts) >= 14:
                return 7    # أسبوعي
            else:
                return max(1, len(ts) // 4)  # ربع البيانات
        except:
            return 7  # القيمة الافتراضية
    
    def _decompose_with_statsmodels(self, ts: pd.Series, period: int) -> Optional[Dict[str, Any]]:
        """التحليل الموسمي باستخدام statsmodels"""
        try:
            self.logger.info("📊 استخدام statsmodels للتحليل الموسمي...")
            
            # التأكد من وجود بيانات كافية
            if len(ts) < period * 2:
                self.logger.warning("⚠️ البيانات غير كافية للتحليل الموسمي")
                return None
            
            # إجراء التحليل الموسمي
            try:
                decomposition = seasonal_decompose(ts, model='additive', period=period)
            except:
                decomposition = seasonal_decompose(ts, model='multiplicative', period=period)
            
            # استخراج المكونات
            result = {
                'method': 'statsmodels',
                'period': period,
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'resid': decomposition.resid,
                'components': {
                    'trend': decomposition.trend.dropna().values.tolist() if decomposition.trend is not None else [],
                    'seasonal': decomposition.seasonal.dropna().values.tolist() if decomposition.seasonal is not None else [],
                    'residual': decomposition.resid.dropna().values.tolist() if decomposition.resid is not None else []
                }
            }
            
            # حساب إحصائيات الموسمية
            if decomposition.seasonal is not None and not decomposition.seasonal.isna().all():
                seasonal_values = decomposition.seasonal.dropna()
                result['seasonal_stats'] = {
                    'mean': float(seasonal_values.mean()),
                    'std': float(seasonal_values.std()),
                    'min': float(seasonal_values.min()),
                    'max': float(seasonal_values.max()),
                    'amplitude': float(seasonal_values.max() - seasonal_values.min())
                }
            else:
                result['seasonal_stats'] = {
                    'mean': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'amplitude': 0
                }
            
            self.logger.info("✅ تم إكمال التحليل الموسمي باستخدام statsmodels")
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ فشل تحليل statsmodels: {str(e)}")
            return None
    
    def _decompose_alternative(self, ts: pd.Series, period: int) -> Dict[str, Any]:
        """
        تحليل موسمي بديل بدون statsmodels
        
        يستخدم المتوسط المتحرك والتجزئة البسيطة
        """
        try:
            self.logger.info("📊 استخدام طريقة التحليل الموسمي البديلة...")
            
            # تحويل إلى array
            y = ts.values
            
            # 1. حساب الاتجاه (Trend) باستخدام المتوسط المتحرك
            if len(y) >= period:
                # استخدام المتوسط المتحرك المركزي
                trend = pd.Series(y).rolling(window=period, center=True).mean()
                trend = trend.fillna(method='bfill').fillna(method='ffill').values
            else:
                trend = np.full_like(y, np.mean(y))
            
            # 2. حساب المكون الموسمي (Seasonal)
            # إزالة الاتجاه
            detrended = y - trend
            
            # حساب المتوسط لكل فترة
            seasonal = np.zeros_like(y)
            for i in range(period):
                indices = np.arange(i, len(y), period)
                if len(indices) > 0:
                    seasonal[indices] = np.mean(detrended[indices])
            
            # جعل المجموع صفراً (تطبيع)
            seasonal = seasonal - np.mean(seasonal)
            
            # 3. حساب المتبقي (Residual)
            residual = y - trend - seasonal
            
            # 4. إنشاء النتائج
            result = {
                'method': 'alternative',
                'period': period,
                'observed': ts,
                'trend': pd.Series(trend, index=ts.index),
                'seasonal': pd.Series(seasonal, index=ts.index),
                'resid': pd.Series(residual, index=ts.index),
                'components': {
                    'trend': trend.tolist(),
                    'seasonal': seasonal.tolist(),
                    'residual': residual.tolist()
                },
                'seasonal_stats': {
                    'mean': float(np.mean(seasonal)),
                    'std': float(np.std(seasonal)),
                    'min': float(np.min(seasonal)),
                    'max': float(np.max(seasonal)),
                    'amplitude': float(np.max(seasonal) - np.min(seasonal))
                },
                'note': 'تم استخدام طريقة بديلة (statsmodels غير متوفر)'
            }
            
            self.logger.info("✅ تم إكمال التحليل الموسمي بالطريقة البديلة")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في التحليل الموسمي البديل: {str(e)}")
            return self._simple_result(e)
    
    def _simple_result(self, error: Exception = None) -> Dict[str, Any]:
        """إرجاع نتائج بسيطة عند حدوث خطأ"""
        return {
            'method': 'simple',
            'period': 7,
            'observed': pd.Series(),
            'trend': pd.Series(),
            'seasonal': pd.Series(),
            'resid': pd.Series(),
            'components': {
                'trend': [],
                'seasonal': [],
                'residual': []
            },
            'seasonal_stats': {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'amplitude': 0
            },
            'note': f'تحليل موسمي مبسط (خطأ: {str(error) if error else "بيانات غير كافية"})'
        }
    
    def get_seasonal_pattern(self, result: Dict[str, Any]) -> pd.DataFrame:
        """
        استخراج النمط الموسمي
        
        Args:
            result: نتائج التحليل الموسمي
            
        Returns:
            DataFrame يحتوي على النمط الموسمي
        """
        try:
            if 'seasonal' not in result:
                return pd.DataFrame()
            
            seasonal = result['seasonal']
            if seasonal is None or seasonal.empty:
                return pd.DataFrame()
            
            # حساب النمط لكل فترة
            period = result.get('period', 7)
            pattern = []
            
            for i in range(period):
                indices = np.arange(i, len(seasonal), period)
                if len(indices) > 0 and len(indices) < len(seasonal):
                    try:
                        avg = seasonal.iloc[indices].mean()
                        pattern.append({
                            'period': i,
                            'value': float(avg) if not np.isnan(avg) else 0,
                            'count': len(indices)
                        })
                    except:
                        continue
            
            return pd.DataFrame(pattern)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في استخراج النمط الموسمي: {str(e)}")
            return pd.DataFrame()