# rfm_analyzer.py
"""
نظام متكامل لتحليل جودة البيانات و RFM مع اختبارات صحة رياضية
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

# ------------------- الجزء الأول: مدقق الجودة (كما سبق) -------------------
class DataValidator:
    """فئة للتحقق من جودة البيانات (مختصرة هنا للاكتفاء بالوظائف الأساسية)"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {
            'total_rows': len(df),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'invalid_dates': 0,
            'negative_values': 0,
            'invalid_customer_ids': 0,
        }
        self._run_checks()

    def _run_checks(self):
        # التحقق من التواريخ (إن وجدت)
        date_cols = [c for c in self.df.columns if 'date' in c.lower()]
        for col in date_cols:
            try:
                invalid = pd.to_datetime(self.df[col], errors='coerce').isnull().sum()
                self.report['invalid_dates'] += int(invalid)
            except:
                pass
        # القيم السالبة في الأعمدة الرقمية التي تحتوي على amount/price
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if any(k in col.lower() for k in ['amount', 'price', 'value']):
                self.report['negative_values'] += int((self.df[col] < 0).sum())
        # معرفات العملاء الفارغة
        cust_cols = [c for c in self.df.columns if 'customer' in c.lower() or 'client' in c.lower()]
        for col in cust_cols:
            self.report['invalid_customer_ids'] += int(self.df[col].isnull().sum())

    def print_report(self):
        r = self.report
        print("\n" + "=" * 50)
        print("📊  DATA QUALITY REPORT (summary)".center(50))
        print("=" * 50)
        print(f"  Rows              : {r['total_rows']}")
        print(f"  Missing values    : {r['missing_values']}")
        print(f"  Duplicates        : {r['duplicates']}")
        print(f"  Invalid dates     : {r['invalid_dates']}")
        print(f"  Negative values   : {r['negative_values']}")
        print(f"  Invalid Customer IDs: {r['invalid_customer_ids']}")
        print("=" * 50 + "\n")


# ------------------- الجزء الثاني: محلل RFM -------------------
class RFMAnalyzer:
    """
    تحليل RFM (Recency, Frequency, Monetary) لكل عميل.
    يفترض وجود أعمدة: Customer, Date, Amount.
    """

    def __init__(self, df: pd.DataFrame, customer_col: str = 'Customer',
                 date_col: str = 'Date', amount_col: str = 'Amount'):
        self.df = df.copy()
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        # تحويل التاريخ
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        # المرجع الزمني (أحدث تاريخ في البيانات)
        self.max_date = self.df[date_col].max()
        self.rfm_table = None
        self._compute_rfm()

    def _compute_rfm(self):
        """حساب مقاييس RFM لكل عميل"""
        # تجميع حسب العميل
        grouped = self.df.groupby(self.customer_col)
        # Frequency = عدد المعاملات
        freq = grouped.size().rename('Frequency')
        # Monetary = مجموع المبالغ
        monetary = grouped[self.amount_col].sum().rename('Monetary')
        # Recency = عدد الأيام منذ آخر معاملة (بالنسبة لأحدث تاريخ في البيانات)
        recency = (self.max_date - grouped[self.date_col].max()).dt.days.rename('Recency')
        # دمج النتائج
        self.rfm_table = pd.concat([recency, freq, monetary], axis=1).reset_index()

    def get_rfm_table(self) -> pd.DataFrame:
        """إرجاع جدول RFM"""
        return self.rfm_table

    def print_rfm(self):
        """طباعة جدول RFM بشكل منسق"""
        print("\n" + "=" * 50)
        print("📈  RFM ANALYSIS RESULTS".center(50))
        print("=" * 50)
        print(self.rfm_table.to_string(index=False))
        print("=" * 50 + "\n")


# ------------------- الجزء الثالث: اختبارات الصحة الرياضية -------------------
def run_validation_tests():
    """
    اختبارات للتأكد من صحة الحسابات باستخدام مجموعة بيانات معروفة.
    """
    print("\n🔬 RUNNING VALIDATION TESTS...")
    # بيانات الاختبار
    data = {
        'Customer': ['A', 'A', 'B'],
        'Date': ['2026-01-01', '2026-01-10', '2026-01-05'],
        'Amount': [100, 200, 500]
    }
    df_test = pd.DataFrame(data)

    # 1. التحقق من جودة البيانات (لا مشاكل متوقعة)
    validator = DataValidator(df_test)
    validator.print_report()

    # 2. تحليل RFM
    rfm = RFMAnalyzer(df_test)
    rfm.print_rfm()

    # 3. التحقق من النتائج المتوقعة
    rfm_table = rfm.get_rfm_table()
    # نتائج متوقعة
    expected = {
        'A': {'Frequency': 2, 'Monetary': 300},
        'B': {'Frequency': 1, 'Monetary': 500}
    }

    passed = True
    for _, row in rfm_table.iterrows():
        cust = row['Customer']
        freq = row['Frequency']
        mon = row['Monetary']
        if cust in expected:
            exp_freq = expected[cust]['Frequency']
            exp_mon = expected[cust]['Monetary']
            if freq != exp_freq or mon != exp_mon:
                print(f"❌ خطأ للعميل {cust}: التردد {freq} (متوقع {exp_freq}), النقدية {mon} (متوقع {exp_mon})")
                passed = False
            else:
                print(f"✅ العميل {cust}: التردد = {freq}, النقدية = {mon} (صحيح)")
        else:
            print(f"⚠️ العميل {cust} غير متوقع في الاختبارات")

    if passed:
        print("\n🎉 جميع الاختبارات اجتازت بنجاح! الحسابات صحيحة رياضياً.")
    else:
        print("\n❌ فشلت بعض الاختبارات، راجع المنطق.")

    return passed


# ------------------- الجزء الرئيسي -------------------
if __name__ == "__main__":
    # تشغيل اختبارات الصحة
    run_validation_tests()

    # مثال على استخدام البيانات الحقيقية (يمكنك تعديله)
    # df_real = pd.read_csv("your_data.csv")
    # validator = DataValidator(df_real)
    # validator.print_report()
    # rfm = RFMAnalyzer(df_real)
    # rfm.print_rfm()
