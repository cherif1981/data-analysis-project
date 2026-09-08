# create_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_data():
    """إنشاء بيانات عينة للمشروع"""
    
    # إعداد البيانات
    np.random.seed(42)
    num_customers = 50
    num_transactions = 200
    
    # إنشاء العملاء
    customer_ids = [f'C{str(i).zfill(3)}' for i in range(1, num_customers + 1)]
    
    # إنشاء التواريخ
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 9, 8)
    date_range = (end_date - start_date).days
    
    # إنشاء البيانات
    data = []
    categories = ['Electronics', 'Books', 'Clothing', 'Furniture', 'Food', 'Sports']
    
    for _ in range(num_transactions):
        customer = random.choice(customer_ids)
        days_offset = random.randint(0, date_range)
        purchase_date = start_date + timedelta(days=days_offset)
        amount = round(random.uniform(20, 1500), 2)
        category = random.choice(categories)
        
        data.append({
            'customer_id': customer,
            'purchase_date': purchase_date.strftime('%Y-%m-%d'),
            'amount': amount,
            'product_category': category
        })
    
    # إنشاء DataFrame
    df = pd.DataFrame(data)
    
    # ترتيب حسب التاريخ
    df = df.sort_values('purchase_date')
    
    # حفظ الملف
    df.to_csv('data/sales.csv', index=False)
    print(f"✅ تم إنشاء {len(df)} سجل في data/sales.csv")
    print(f"📊 عدد العملاء: {df['customer_id'].nunique()}")
    print(f"💰 إجمالي المبيعات: ${df['amount'].sum():,.2f}")
    print(f"📅 فترة البيانات: من {df['purchase_date'].min()} إلى {df['purchase_date'].max()}")
    
    return df

if __name__ == "__main__":
    # التأكد من وجود مجلد data
    import os
    os.makedirs('data', exist_ok=True)
    
    # إنشاء البيانات
    df = create_sample_data()
    
    # عرض عينة من البيانات
    print("\n📋 عينة من البيانات:")
    print(df.head(10))