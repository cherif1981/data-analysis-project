# src/features/engineering.py
"""
Feature Engineering - هندسة الميزات المتقدمة
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    إنشاء وتحويل الميزات المتقدمة
    """
    
    def __init__(self):
        self.feature_history = []
        logger.info("تم تهيئة مهندس الميزات")
    
    def create_polynomial_features(self, X: pd.DataFrame, 
                                   columns: List[str], degree: int = 2) -> pd.DataFrame:
        """
        إنشاء ميزات متعددة الحدود
        """
        logger.info(f"إنشاء ميزات متعددة الحدود (degree={degree})...")
        X_new = X.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                new_col = f"{col}_pow{d}"
                X_new[new_col] = X[col] ** d
                self.feature_history.append(new_col)
        
        logger.info(f"✅ تم إنشاء {len(self.feature_history)} ميزة جديدة")
        return X_new
    
    def create_interaction_features(self, X: pd.DataFrame, 
                                   columns: List[List[str]]) -> pd.DataFrame:
        """
        إنشاء ميزات التفاعل بين متغيرين
        """
        logger.info(f"إنشاء ميزات التفاعل...")
        X_new = X.copy()
        interaction_count = 0
        
        for col_pair in columns:
            if len(col_pair) == 2:
                new_col = f"{col_pair[0]}_{col_pair[1]}_interaction"
                X_new[new_col] = X[col_pair[0]] * X[col_pair[1]]
                self.feature_history.append(new_col)
                interaction_count += 1
        
        logger.info(f"✅ تم إنشاء {interaction_count} ميزة تفاعل")
        return X_new
    
    def create_binning_features(self, X: pd.DataFrame, 
                               columns: Dict[str, int]) -> pd.DataFrame:
        """
        تقسيم الميزات الرقمية إلى فئات
        """
        logger.info(f"إنشاء ميزات التقسيم (Binning)...")
        X_new = X.copy()
        
        for col, n_bins in columns.items():
            new_col = f"{col}_binned"
            X_new[new_col] = pd.cut(X[col], bins=n_bins, labels=False)
            self.feature_history.append(new_col)
        
        logger.info(f"✅ تم إنشاء {len(columns)} ميزة مقسمة")
        return X_new
    
    def create_ratio_features(self, X: pd.DataFrame, 
                             ratios: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        إنشاء ميزات النسب
        """
        logger.info(f"إنشاء ميزات النسب...")
        X_new = X.copy()
        
        for numerator, denominator in ratios:
            if denominator in X.columns and (X[denominator] != 0).all():
                new_col = f"{numerator}_to_{denominator}_ratio"
                X_new[new_col] = X[numerator] / X[denominator]
                self.feature_history.append(new_col)
        
        logger.info(f"✅ تم إنشاء {len(self.feature_history)} ميزة نسبة")
        return X_new
    
    def create_statistical_features(self, X: pd.DataFrame, 
                                   group_col: str, agg_cols: List[str]) -> pd.DataFrame:
        """
        إنشاء ميزات إحصائية مجمعة
        """
        logger.info(f"إنشاء ميزات إحصائية...")
        X_new = X.copy()
        
        # حساب الإحصائيات المجمعة
        stats = X.groupby(group_col)[agg_cols].agg(['mean', 'std', 'min', 'max'])
        
        for col in agg_cols:
            X_new[f'{col}_group_mean'] = X[group_col].map(stats[col]['mean'])
            X_new[f'{col}_group_std'] = X[group_col].map(stats[col]['std'])
        
        logger.info(f"✅ تم إنشاء ميزات إحصائية")
        return X_new
    
    def get_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص الميزات المنشأة"""
        return {
            'total_features_created': len(self.feature_history),
            'features': self.feature_history
        }
