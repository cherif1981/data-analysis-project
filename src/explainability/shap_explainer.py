# src/explainability/shap_explainer.py
"""
SHAP - Explainability Module - شرح نتائج النموذج
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    شرح قرارات النموذج باستخدام SHAP
    """
    
    def __init__(self, model: Any, X_background: np.ndarray):
        """
        Args:
            model: النموذج المدرب
            X_background: عينات خلفية للشرح (عادة 10-20% من البيانات)
        """
        self.model = model
        self.X_background = X_background
        self.explainer = None
        logger.info("تم تهيئة شارح SHAP")
    
    def create_tree_explainer(self) -> 'SHAPExplainer':
        """
        إنشاء شارح SHAP لنماذج الأشجار
        """
        logger.info("📊 إنشاء شارح SHAP للأشجار...")
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("✅ تم إنشاء شارح SHAP")
        return self
    
    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        حساب قيم SHAP للعينات
        """
        logger.info("حساب قيم SHAP...")
        shap_values = self.explainer.shap_values(X)
        logger.info(f"✅ تم حساب قيم SHAP (الشكل: {np.array(shap_values).shape})")
        return shap_values
    
    def get_feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        الحصول على أهمية الميزات من قيم SHAP
        """
        # حساب متوسط القيمة المطلقة لـ SHAP
        if isinstance(shap_values, list):
            # للتصنيف متعدد الفئات
            mean_abs_shap = np.mean(np.abs(shap_values[0]), axis=0)
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': range(len(mean_abs_shap)),
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        logger.info(f"✅ أهم 5 ميزات:\n{importance_df.head()}")
        return importance_df
