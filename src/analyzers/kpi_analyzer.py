class KPIAnalyzer:
    def calculate(self, df):
        kpis = {
            'total_revenue': df['amount'].sum(),
            'average_order_value': df['amount'].mean(),
            'customer_acquisition_cost': ...,
            'conversion_rate': ...,
            'churn_rate': ...
        }
        return kpis