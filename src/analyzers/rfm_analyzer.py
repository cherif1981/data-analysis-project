# src/analyzers/rfm_analyzer.py - النسخة المصححة بالكامل
"""
تحليل RFM (Recency, Frequency, Monetary)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, Optional

class RFMAnalyzer:
    """تحليل RFM لتقسيم العملاء"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_rfm(self, df: pd.DataFrame, reference_date: Optional[str] = None) -> pd.DataFrame:
        """
        حساب مقاييس RFM لكل عميل
        
        Args:
            df: DataFrame يحتوي على بيانات المعاملات
            reference_date: تاريخ المرجع (اختياري)
            
        Returns:
            DataFrame يحتوي على مقاييس RFM لكل عميل
        """
        try:
            self.logger.info("📊 بدء حساب RFM...")
            
            # التحقق من وجود البيانات
            if df.empty:
                raise ValueError("البيانات فارغة")
            
            # عرض معلومات البيانات للتصحيح
            self.logger.info(f"📋 أعمدة البيانات: {df.columns.tolist()}")
            self.logger.info(f"📊 عدد الصفوف: {len(df)}")
            
            # التأكد من وجود الأعمدة المطلوبة
            required_columns = ['customer_id', 'purchase_date', 'amount']
            missing_columns = []
            
            for col in required_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                self.logger.error(f"❌ الأعمدة المفقودة: {missing_columns}")
                self.logger.info("🔍 محاولة إيجاد أعمدة بديلة...")
                
                # محاولة إيجاد أعمدة بديلة
                df = self._find_alternative_columns(df)
                
                # التحقق مرة أخرى
                still_missing = [col for col in required_columns if col not in df.columns]
                if still_missing:
                    raise ValueError(f"لا يمكن العثور على الأعمدة المطلوبة: {still_missing}")
            
            # تحويل تاريخ الشراء إلى datetime
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            
            # تحديد تاريخ المرجع
            if reference_date is None:
                reference_date = df['purchase_date'].max() + pd.Timedelta(days=1)
            else:
                reference_date = pd.to_datetime(reference_date)
            
            self.logger.info(f"📅 تاريخ المرجع: {reference_date}")
            
            # ✅ حساب RFM - الطريقة الصحيحة مع groupby
            # التأكد من أن الأعمدة موجودة قبل groupby
            if 'customer_id' not in df.columns:
                raise KeyError("العمود 'customer_id' غير موجود في البيانات")
            
            # حساب RFM باستخدام groupby
            rfm = df.groupby('customer_id').agg({
                'purchase_date': lambda x: (reference_date - x.max()).days,
                'customer_id': 'count',  # هذا سيعطي عدد المعاملات
                'amount': 'sum'
            }).rename(columns={
                'purchase_date': 'recency',
                'customer_id': 'frequency',
                'amount': 'monetary'
            })
            
            self.logger.info(f"✅ تم حساب RFM لـ {len(rfm)} عميل")
            
            # عرض إحصائيات سريعة
            self.logger.info(f"📊 متوسط Recency: {rfm['recency'].mean():.1f} يوم")
            self.logger.info(f"📊 متوسط Frequency: {rfm['frequency'].mean():.1f} طلب")
            self.logger.info(f"📊 متوسط Monetary: ${rfm['monetary'].mean():.2f}")
            
            return rfm
            
        except KeyError as e:
            self.logger.error(f"❌ خطأ في المفتاح: {str(e)}")
            self.logger.info("📋 الأعمدة المتوفرة: " + ", ".join(df.columns.tolist()))
            raise
        except Exception as e:
            self.logger.error(f"❌ خطأ في حساب RFM: {str(e)}")
            raise
    
    def _find_alternative_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        البحث عن أعمدة بديلة إذا كانت الأسماء مختلفة
        
        Args:
            df: DataFrame للبحث
            
        Returns:
            DataFrame مع أعمدة موحدة
        """
        # قاموس لتخزين الأعمدة البديلة
        column_mapping = {}
        
        # البحث عن عمود customer_id
        customer_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                        ['customer', 'client', 'user', 'id', 'cust'])]
        if customer_cols:
            # اختيار العمود الأكثر احتمالاً
            best_col = customer_cols[0]
            if 'id' in best_col.lower() or 'customer' in best_col.lower() or 'client' in best_col.lower():
                column_mapping[best_col] = 'customer_id'
                self.logger.info(f"✅ تم تعيين {best_col} كـ customer_id")
        
        # البحث عن عمود purchase_date
        date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                    ['date', 'day', 'purchase', 'order', 'transaction', 'time'])]
        if date_cols:
            best_col = date_cols[0]
            column_mapping[best_col] = 'purchase_date'
            self.logger.info(f"✅ تم تعيين {best_col} كـ purchase_date")
        
        # البحث عن عمود amount
        amount_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                      ['amount', 'price', 'total', 'value', 'sales', 'revenue', 'cost'])]
        if amount_cols:
            best_col = amount_cols[0]
            column_mapping[best_col] = 'amount'
            self.logger.info(f"✅ تم تعيين {best_col} كـ amount")
        
        # إعادة تسمية الأعمدة
        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.info(f"✅ تم إعادة تسمية الأعمدة: {column_mapping}")
        
        # إذا لم يتم العثور على الأعمدة المطلوبة، قم بإنشائها
        if 'customer_id' not in df.columns:
            self.logger.warning("⚠️ لم يتم العثور على عمود customer_id، سيتم إنشاؤه")
            df['customer_id'] = [f'C{str(i).zfill(3)}' for i in range(1, len(df) + 1)]
        
        if 'purchase_date' not in df.columns:
            self.logger.warning("⚠️ لم يتم العثور على عمود purchase_date، سيتم إنشاؤه")
            from datetime import datetime, timedelta
            import random
            start_date = datetime(2026, 1, 1)
            df['purchase_date'] = [
                (start_date + timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')
                for _ in range(len(df))
            ]
        
        if 'amount' not in df.columns:
            self.logger.warning("⚠️ لم يتم العثور على عمود amount، سيتم إنشاؤه")
            df['amount'] = np.random.uniform(20, 1500, len(df)).round(2)
        
        return df
    
    def assign_segments(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        تقسيم العملاء إلى شرائح بناءً على RFM
        
        Args:
            rfm_df: DataFrame يحتوي على مقاييس RFM
            
        Returns:
            DataFrame مع إضافة عمود الشريحة
        """
        try:
            if rfm_df.empty:
                raise ValueError("بيانات RFM فارغة")
            
            # نسخ DataFrame لتجنب التعديل على الأصل
            rfm_segmented = rfm_df.copy()
            
            # التأكد من وجود الأعمدة المطلوبة
            required_cols = ['recency', 'frequency', 'monetary']
            for col in required_cols:
                if col not in rfm_segmented.columns:
                    raise KeyError(f"العمود {col} غير موجود في بيانات RFM")
            
            # حساب الدرجات
            try:
                # تقسيم إلى 4 مجموعات متساوية
                rfm_segmented['r_score'] = pd.qcut(
                    rfm_segmented['recency'], 
                    q=4, 
                    labels=['4', '3', '2', '1'],
                    duplicates='drop'
                )
                
                rfm_segmented['f_score'] = pd.qcut(
                    rfm_segmented['frequency'], 
                    q=4, 
                    labels=['1', '2', '3', '4'],
                    duplicates='drop'
                )
                
                rfm_segmented['m_score'] = pd.qcut(
                    rfm_segmented['monetary'], 
                    q=4, 
                    labels=['1', '2', '3', '4'],
                    duplicates='drop'
                )
                
            except ValueError as e:
                self.logger.warning(f"⚠️ مشكلة في التقسيم الرباعي: {str(e)}")
                self.logger.info("📊 استخدام التقسيم اليدوي بدلاً من ذلك...")
                
                # استخدام التقسيم اليدوي
                rfm_segmented['r_score'] = pd.cut(
                    rfm_segmented['recency'],
                    bins=4,
                    labels=['4', '3', '2', '1']
                )
                
                rfm_segmented['f_score'] = pd.cut(
                    rfm_segmented['frequency'],
                    bins=4,
                    labels=['1', '2', '3', '4']
                )
                
                rfm_segmented['m_score'] = pd.cut(
                    rfm_segmented['monetary'],
                    bins=4,
                    labels=['1', '2', '3', '4']
                )
            
            # دمج الدرجات
            rfm_segmented['rfm_score'] = (
                rfm_segmented['r_score'].astype(str) + 
                rfm_segmented['f_score'].astype(str) + 
                rfm_segmented['m_score'].astype(str)
            )
            
            # تحديد الشرائح
            def get_segment(row):
                try:
                    r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
                    
                    if r >= 3 and f >= 3 and m >= 3:
                        return 'Champions'
                    elif r >= 3 and f >= 2 and m >= 2:
                        return 'Loyal'
                    elif r >= 3 and f >= 1 and m >= 1:
                        return 'Potential'
                    elif r >= 2 and f >= 2 and m >= 2:
                        return 'New'
                    elif r >= 1 and f >= 1 and m >= 1:
                        return 'At Risk'
                    else:
                        return 'Others'
                except:
                    return 'Others'
            
            rfm_segmented['segment'] = rfm_segmented.apply(get_segment, axis=1)
            
            # إضافة customer_id كعمود (إذا كان موجوداً في الفهرس)
            if 'customer_id' not in rfm_segmented.columns:
                if rfm_segmented.index.name == 'customer_id' or rfm_segmented.index.name is None:
                    rfm_segmented['customer_id'] = rfm_segmented.index
            
            self.logger.info(f"✅ تم تقسيم العملاء إلى {rfm_segmented['segment'].nunique()} شرائح")
            
            # إحصائيات الشرائح
            segment_stats = rfm_segmented.groupby('segment').agg({
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean',
                'customer_id': 'count' if 'customer_id' in rfm_segmented.columns else 'recency'
            })
            
            if 'customer_id' in segment_stats.columns:
                segment_stats = segment_stats.rename(columns={'customer_id': 'count'})
            
            self.logger.info("\n📊 إحصائيات الشرائح:")
            for segment, stats in segment_stats.iterrows():
                self.logger.info(f"   {segment}: {stats['count'] if 'count' in stats else 'N/A'} عميل")
            
            return rfm_segmented
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تقسيم الشرائح: {str(e)}")
            raise
    
    def get_segment_recommendations(self, segment: str) -> Dict[str, Any]:
        """الحصول على توصيات لكل شريحة"""
        recommendations = {
            'Champions': {
                'strategy': 'الحفاظ على العملاء المميزين',
                'actions': [
                    'تقديم عروض VIP',
                    'برامج ولاء متقدمة',
                    'طلب تقييمات ومراجعات',
                    'دعوة للمناسبات الخاصة'
                ],
                'priority': 'عالية'
            },
            'Loyal': {
                'strategy': 'تعزيز الولاء',
                'actions': [
                    'برامج مكافآت',
                    'عروض حصرية',
                    'تواصل منتظم',
                    'خصومات على المشتريات المتكررة'
                ],
                'priority': 'عالية'
            },
            'Potential': {
                'strategy': 'تحويل العملاء المحتملين',
                'actions': [
                    'عروض ترحيبية',
                    'توصيات منتجات',
                    'تذكير بالمنتجات التي تم عرضها',
                    'عروض محدودة الوقت'
                ],
                'priority': 'متوسطة'
            },
            'New': {
                'strategy': 'تحويل العملاء الجدد',
                'actions': [
                    'تجربة مجانية',
                    'عروض أول طلب',
                    'دليل استخدام المنتج',
                    'استبيان رضا'
                ],
                'priority': 'متوسطة'
            },
            'At Risk': {
                'strategy': 'استعادة العملاء المهددين',
                'actions': [
                    'عروض استعادة',
                    'استبيان سبب التوقف',
                    'خصومات خاصة',
                    'تواصل شخصي'
                ],
                'priority': 'عالية'
            },
            'Others': {
                'strategy': 'تحليل إضافي',
                'actions': [
                    'دراسة سلوك العميل',
                    'تحليل أسباب عدم التفاعل',
                    'عروض عامة'
                ],
                'priority': 'منخفضة'
            }
        }
        
        return recommendations.get(segment, recommendations['Others'])