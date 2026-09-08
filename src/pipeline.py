# pipeline.py
# نظام خطوط أنابيب تحليل البيانات بالكامل (نسخة عربية مع تعليقات)

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# إعداد نظام التسجيل (logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    فئة رئيسية تحتوي على جميع مراحل تحليل البيانات:
    التحميل، التحقق، التنظيف، هندسة الخصائص، التحليل، RFM، التنبؤ، التقارير، التصدير.
    """
    
    def __init__(self, input_path: str, output_dir: str = "outputs"):
        """
        تهيئة المسارات والحالة الداخلية.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None  # سيحتوي على البيانات أثناء التنفيذ
        self.results = {}  # لتخزين نتائج كل مرحلة
        
        # إنشاء مجلد المخرجات إن لم يكن موجوداً
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"تم تهيئة الأنبوب، سيتم حفظ المخرجات في: {self.output_dir}")
    
    # ----------------------------------------------------------------
    # 1. التحميل (Loading)
    # ----------------------------------------------------------------
    def load_data(self):
        """تحميل البيانات من ملف CSV أو Excel."""
        logger.info(f"بدء تحميل البيانات من: {self.input_path}")
        try:
            if self.input_path.endswith('.csv'):
                self.df = pd.read_csv(self.input_path, encoding='utf-8')
            elif self.input_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.input_path)
            else:
                raise ValueError("نوع الملف غير مدعوم. استخدم CSV أو Excel.")
            
            logger.info(f"تم التحميل بنجاح. عدد الصفوف: {self.df.shape[0]}, عدد الأعمدة: {self.df.shape[1]}")
            return self.df
        except Exception as e:
            logger.error(f"فشل التحميل: {e}")
            sys.exit(1)
    
    # ----------------------------------------------------------------
    # 2. التحقق من الصحة (Validation)
    # ----------------------------------------------------------------
    def validate_data(self):
        """التحقق من البيانات: قيم مفقودة، تكرارات، أنواع البيانات."""
        logger.info("بدء مرحلة التحقق من صحة البيانات...")
        
        # تقرير أولي عن البيانات المفقودة
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        validation_report = pd.DataFrame({
            'العمود': self.df.columns,
            'القيم المفقودة': missing_counts,
            'النسبة المئوية (%)': missing_percent.round(2),
            'النوع': self.df.dtypes.values
        })
        
        # التحقق من الصفوف المكررة
        duplicates = self.df.duplicated().sum()
        logger.info(f"عدد الصفوف المكررة: {duplicates}")
        
        # تخزين التقرير
        self.results['validation'] = validation_report
        logger.info("اكتمل التحقق من الصحة.")
        return validation_report
    
    # ----------------------------------------------------------------
    # 3. التنظيف (Cleaning)
    # ----------------------------------------------------------------
    def clean_data(self):
        """معالجة القيم المفقودة، إزالة المكررات، ومعالجة القيم المتطرفة."""
        logger.info("بدء مرحلة تنظيف البيانات...")
        
        # نسخ احتياطي
        df_clean = self.df.copy()
        
        # 1. إزالة المكررات
        df_clean.drop_duplicates(inplace=True)
        logger.info(f"تم إزالة المكررات. عدد الصفوف الآن: {len(df_clean)}")
        
        # 2. معالجة القيم المفقودة (ملء الأعمدة الرقمية بالمتوسط، والفئوية بالقيمة الأكثر تكراراً)
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                # عددي: نملأ بالمتوسط
                if df_clean[col].isnull().any():
                    mean_val = df_clean[col].mean()
                    df_clean[col].fillna(mean_val, inplace=True)
                    logger.info(f"عمود '{col}': تم ملء القيم المفقودة بالمتوسط ({mean_val:.2f})")
            else:
                # فئوي: نملأ بالقيمة الأكثر تكراراً (mode)
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'مجهول'
                    df_clean[col].fillna(mode_val, inplace=True)
                    logger.info(f"عمود '{col}': تم ملء القيم المفقودة بالقيمة الأكثر تكراراً ('{mode_val}')")
        
        # 3. معالجة القيم المتطرفة (باستخدام IQR للأعمدة الرقمية) - اختيارية
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            if len(outliers) > 0:
                # نقوم بقص (clip) القيم المتطرفة بدلاً من حذفها
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"عمود '{col}': تم قص {len(outliers)} قيمة متطرفة.")
        
        self.df = df_clean
        self.results['cleaning'] = f"تم التنظيف. عدد الصفوف النهائي: {len(self.df)}"
        logger.info("اكتملت مرحلة التنظيف.")
        return self.df
    
    # ----------------------------------------------------------------
    # 4. هندسة الخصائص (Feature Engineering)
    # ----------------------------------------------------------------
    def engineer_features(self):
        """إنشاء خصائص جديدة لتحسين التحليل (مثال: استخراج التاريخ، تقسيم الفئات)."""
        logger.info("بدء مرحلة هندسة الخصائص...")
        
        # البحث عن أعمدة تحتوي على تواريخ (نفترض وجود عمود اسمه 'تاريخ' أو 'date')
        date_cols = [col for col in self.df.columns if 'تاريخ' in col or 'date' in col.lower()]
        if date_cols:
            for col in date_cols:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    # استخراج مكونات التاريخ
                    self.df[f'{col}_سنة'] = self.df[col].dt.year
                    self.df[f'{col}_شهر'] = self.df[col].dt.month
                    self.df[f'{col}_اليوم'] = self.df[col].dt.day
                    self.df[f'{col}_يوم_الأسبوع'] = self.df[col].dt.dayofweek  # 0=اثنين
                    logger.info(f"تم استخراج مكونات التاريخ من عمود '{col}'.")
                except:
                    logger.warning(f"تعذر تحويل عمود '{col}' إلى تاريخ.")
        
        # إنشاء عمود 'الربع' بناءً على الشهر (لتقسيم الفئات)
        if 'تاريخ_شهر' in self.df.columns or 'month' in self.df.columns:
            month_col = 'تاريخ_شهر' if 'تاريخ_شهر' in self.df.columns else 'month'
            self.df['الربع'] = np.ceil(self.df[month_col] / 3).astype(int)
            logger.info("تم إنشاء عمود 'الربع'.")
        
        self.results['features'] = f"أضيفت خصائص جديدة. عدد الأعمدة الآن: {self.df.shape[1]}"
        logger.info("اكتملت هندسة الخصائص.")
        return self.df
    
    # ----------------------------------------------------------------
    # 5. التحليل الاستكشافي (Analytics)
    # ----------------------------------------------------------------
    def run_analytics(self):
        """إحصائيات وصفية وتحليل استكشافي أساسي."""
        logger.info("بدء مرحلة التحليل الإحصائي...")
        
        stats = self.df.describe(include='all').transpose()
        # إضافة نسبة القيم الفريدة
        stats['عدد_فريد'] = self.df.nunique().values
        stats['نسبة_فريد_%'] = (stats['عدد_فريد'] / len(self.df) * 100).round(2)
        
        self.results['analytics'] = stats
        logger.info("اكتمل التحليل الإحصائي.")
        return stats
    
    # ----------------------------------------------------------------
    # 6. تحليل RFM (الحداثة - التكرار - القيمة النقدية)
    # ----------------------------------------------------------------
    def run_rfm(self):
        """
        تحليل RFM كلاسيكي. يفترض وجود أعمدة:
        - 'عميل' أو 'customer_id'
        - 'تاريخ' أو 'date'
        - 'المبلغ' أو 'amount'
        """
        logger.info("بدء تحليل RFM...")
        
        # محاولة اكتشاف الأعمدة المنطقية
        id_col = next((c for c in self.df.columns if 'عميل' in c or 'customer' in c.lower()), None)
        date_col = next((c for c in self.df.columns if 'تاريخ' in c or 'date' in c.lower()), None)
        amount_col = next((c for c in self.df.columns if 'مبلغ' in c or 'amount' in c.lower() or 'سعر' in c), None)
        
        if not id_col or not date_col or not amount_col:
            logger.warning("تعذر العثور على أعمدة RFM (عميل، تاريخ، مبلغ). سيتم تخطي هذه المرحلة.")
            self.results['rfm'] = "لم يتم تنفيذ RFM لعدم توفر الأعمدة المطلوبة."
            return None
        
        # تحويل التاريخ
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        current_date = self.df[date_col].max() + pd.Timedelta(days=1)
        
        # حساب RFM
        rfm = self.df.groupby(id_col).agg({
            date_col: lambda x: (current_date - x.max()).days,  # الحداثة
            id_col: 'count',  # التكرار
            amount_col: 'sum'  # القيمة النقدية
        }).rename(columns={
            date_col: 'الحداثة_أيام',
            id_col: 'التكرار',
            amount_col: 'القيمة_النقدية'
        })
        
        # تقسيم إلى درجات (1-4)
        rfm['درجة_الحداثة'] = pd.qcut(rfm['الحداثة_أيام'], q=4, labels=[4, 3, 2, 1])
        rfm['درجة_التكرار'] = pd.qcut(rfm['التكرار'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
        rfm['درجة_القيمة'] = pd.qcut(rfm['القيمة_النقدية'], q=4, labels=[1, 2, 3, 4])
        
        # حساب النتيجة الإجمالية
        rfm['RFM_النتيجة'] = rfm['درجة_الحداثة'].astype(str) + rfm['درجة_التكرار'].astype(str) + rfm['درجة_القيمة'].astype(str)
        
        self.results['rfm'] = rfm
        logger.info(f"اكتمل تحليل RFM لعدد {len(rfm)} عميل.")
        return rfm
    
    # ----------------------------------------------------------------
    # 7. التنبؤ (Forecasting) - نموذج مبسط (المتوسط المتحرك)
    # ----------------------------------------------------------------
    def run_forecasting(self):
        """تنبؤ بسيط باستخدام المتوسط المتحرك أو نموذج ARIMA وهمي."""
        logger.info("بدء مرحلة التنبؤ...")
        
        # البحث عن عمود رقمي للتنبؤ (المبيعات مثلاً)
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            logger.warning("لا يوجد أعمدة رقمية للتنبؤ. سيتم تخطي المرحلة.")
            self.results['forecasting'] = "لا توجد بيانات رقمية للتنبؤ."
            return None
        
        target_col = num_cols[0]  # نأخذ أول عمود رقمي
        logger.info(f"سيتم التنبؤ باستخدام العمود: '{target_col}'")
        
        # إذا كان هناك تاريخ، نستخدمه كفهرس
        date_col = next((c for c in self.df.columns if 'تاريخ' in c or 'date' in c.lower()), None)
        if date_col:
            try:
                ts_data = self.df.set_index(date_col)[target_col].sort_index()
                # متوسط متحرك بسيط لـ 3 فترات
                forecast_series = ts_data.rolling(window=3).mean().iloc[-5:]  # آخر 5 قيم متوقعة
                forecast = forecast_series.fillna(ts_data.mean()).values.tolist()
            except:
                forecast = [self.df[target_col].mean()] * 5
        else:
            # تنبؤ ثابت بناءً على المتوسط العام
            forecast = [self.df[target_col].mean()] * 5
        
        self.results['forecasting'] = {
            'العمود_المستهدف': target_col,
            'التنبؤ_للفترات_القادمة': forecast,
            'ملاحظة': 'تنبؤ مبني على المتوسط المتحرك (افتراضي)'
        }
        logger.info(f"اكتمل التنبؤ. النتائج: {forecast}")
        return self.results['forecasting']
    
    # ----------------------------------------------------------------
    # 8. التقارير (Reporting)
    # ----------------------------------------------------------------
    def generate_report(self):
        """إنشاء تقرير نصي شامل يلخص جميع المراحل."""
        logger.info("بدء إنشاء التقرير النهائي...")
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("تقرير تحليل البيانات الشامل")
        report_lines.append(f"تاريخ التنفيذ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append(f"إجمالي الصفوف: {len(self.df)}")
        report_lines.append(f"إجمالي الأعمدة: {self.df.shape[1]}")
        
        # إضافة نتائج كل مرحلة
        if 'validation' in self.results:
            report_lines.append("\n--- نتائج التحقق ---")
            report_lines.append(self.results['validation'].to_string())
        
        if 'cleaning' in self.results:
            report_lines.append(f"\n--- التنظيف ---\n{self.results['cleaning']}")
        
        if 'rfm' in self.results and isinstance(self.results['rfm'], pd.DataFrame):
            report_lines.append("\n--- أفضل 5 عملاء RFM ---")
            report_lines.append(self.results['rfm'].head(5).to_string())
        
        if 'forecasting' in self.results and isinstance(self.results['forecasting'], dict):
            report_lines.append(f"\n--- التنبؤ ---\n{self.results['forecasting']}")
        
        report_text = "\n".join(report_lines)
        
        # حفظ التقرير في ملف
        report_path = os.path.join(self.output_dir, "تقرير_التحليل.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.results['report'] = report_path
        logger.info(f"تم حفظ التقرير في: {report_path}")
        return report_text
    
    # ----------------------------------------------------------------
    # 9. التصدير (Export)
    # ----------------------------------------------------------------
    def export_data(self):
        """تصدير البيانات النهائية ونتائج RFM إلى ملفات CSV/Excel."""
        logger.info("بدء تصدير البيانات...")
        
        # 1. تصدير البيانات النظيفة
        clean_path = os.path.join(self.output_dir, "البيانات_النظيفة.csv")
        self.df.to_csv(clean_path, index=False, encoding='utf-8-sig')
        logger.info(f"تم تصدير البيانات النظيفة إلى: {clean_path}")
        
        # 2. تصدير نتائج RFM إن وجدت
        if 'rfm' in self.results and isinstance(self.results['rfm'], pd.DataFrame):
            rfm_path = os.path.join(self.output_dir, "نتائج_RFM.csv")
            self.results['rfm'].to_csv(rfm_path, index=True, encoding='utf-8-sig')
            logger.info(f"تم تصدير نتائج RFM إلى: {rfm_path}")
        
        # 3. تصدير الإحصائيات
        if 'analytics' in self.results and isinstance(self.results['analytics'], pd.DataFrame):
            stats_path = os.path.join(self.output_dir, "الإحصائيات_الوصفية.csv")
            self.results['analytics'].to_csv(stats_path, index=True, encoding='utf-8-sig')
            logger.info(f"تم تصدير الإحصائيات إلى: {stats_path}")
        
        self.results['export'] = "تم التصدير بنجاح."
        logger.info("اكتملت عملية التصدير.")
    
    # ----------------------------------------------------------------
    # تشغيل الأنبوب بالكامل (Run Full Pipeline)
    # ----------------------------------------------------------------
    def run_pipeline(self, steps: list = None):
        """
        تشغيل تسلسل المراحل.
        إذا لم يتم تحديد steps، يتم تشغيل الكل.
        """
        if steps is None:
            steps = ['load', 'validate', 'clean', 'feature', 'analytics', 'rfm', 'forecast', 'report', 'export']
        
        logger.info(f"بدء تشغيل الأنبوب بالمراحل: {steps}")
        
        # خريطة المراحل
        stage_map = {
            'load': self.load_data,
            'validate': self.validate_data,
            'clean': self.clean_data,
            'feature': self.engineer_features,
            'analytics': self.run_analytics,
            'rfm': self.run_rfm,
            'forecast': self.run_forecasting,
            'report': self.generate_report,
            'export': self.export_data
        }
        
        for step in steps:
            if step in stage_map:
                logger.info(f"--- تنفيذ المرحلة: {step.upper()} ---")
                stage_map[step]()
            else:
                logger.warning(f"المرحلة '{step}' غير معروفة. سيتم تخطيها.")
        
        logger.info("✅ اكتمل تشغيل الأنبوب بنجاح!")


# ----------------------------------------------------------------
# نقطة الدخول الرئيسية (سطر الأوامر)
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="أنبوب تحليل البيانات - Data Pipeline")
    parser.add_argument("--input", "-i", type=str, default="data/sample_data.csv",
                        help="مسار ملف البيانات المدخل (CSV أو Excel)")
    parser.add_argument("--output", "-o", type=str, default="outputs",
                        help="مجلد حفظ المخرجات")
    parser.add_argument("--step", "-s", type=str, nargs='+',
                        choices=['load', 'validate', 'clean', 'feature', 'analytics', 'rfm', 'forecast', 'report', 'export'],
                        default=None, help="تحديد مراحل محددة للتشغيل (افتراضي: الكل)")
    
    args = parser.parse_args()
    
    # إنشاء كائن الأنبوب وتشغيله
    pipeline = DataPipeline(input_path=args.input, output_dir=args.output)
    pipeline.run_pipeline(steps=args.step)