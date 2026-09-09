# src/preprocessing/pipeline.py
"""
Preprocessing Pipeline - بناء خط أنابيب المعالجة المسبقة
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    معالجة البيانات: تقسيم البيانات، ترميز، تطبيع
    """
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2, val_size: float = 0.1):
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.scaler = None
        self.label_encoders = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_names = []
        logger.info("تم تهيئة خط أنابيب المعالجة المسبقة")
    
    def identify_features(self, df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
        """
        تحديد الأعمدة الرقمية والفئوية
        """
        logger.info(f"تحديد أنواع الميزات...")
        
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # إزالة العمود المستهدف من القائمة
        if target_col in categorical:
            categorical.remove(target_col)
        if target_col in numerical:
            numerical.remove(target_col)
        
        self.categorical_features = categorical
        self.numerical_features = numerical
        self.feature_names = categorical + numerical
        
        logger.info(f"✅ الميزات الرقمية ({len(numerical)}): {numerical}")
        logger.info(f"✅ الميزات الفئوية ({len(categorical)}): {categorical}")
        
        return numerical, categorical
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.Series, pd.Series, pd.Series
    ]:
        """
        فصل البيانات: Train / Validation / Test
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"فصل البيانات إلى Train/Validation/Test...")
        
        # أولاً: فصل Test (80% train+val, 20% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if y.dtype == 'object' or len(y.unique()) < 20 else None
        )
        
        # ثانياً: فصل Validation من Train (90% train, 10% val من الـ 80%)
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp if y_temp.dtype == 'object' or len(y_temp.unique()) < 20 else None
        )
        
        logger.info(f"📊 حجم البيانات:")
        logger.info(f"   Train: {len(X_train)} عينة ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"   Validation: {len(X_val)} عينة ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"   Test: {len(X_test)} عينة ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encode_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                       X_test: pd.DataFrame, categorical_cols: List[str]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        ترميز الميزات الفئوية باستخدام Label Encoding
        """
        logger.info(f"ترميز الميزات الفئوية...")
        
        X_train_encoded = X_train.copy()
        X_val_encoded = X_val.copy()
        X_test_encoded = X_test.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            X_val_encoded[col] = le.transform(X_val[col].astype(str))
            X_test_encoded[col] = le.transform(X_test[col].astype(str))
            
            self.label_encoders[col] = le
            logger.info(f"✅ تم ترميز '{col}' ({len(le.classes_)} فئات)")
        
        return X_train_encoded, X_val_encoded, X_test_encoded
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                      X_test: np.ndarray, numerical_cols: List[str]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        تطبيع الميزات الرقمية باستخدام StandardScaler
        """
        logger.info(f"تطبيع الميزات الرقمية...")
        
        if len(numerical_cols) == 0:
            logger.warning("لا توجد ميزات رقمية للتطبيع")
            return X_train, X_val, X_test
        
        self.scaler = StandardScaler()
        
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = self.scaler.transform(X_val[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        logger.info(f"✅ تم تطبيع {len(numerical_cols)} ميزة رقمية")
        
        return X_train, X_val, X_test
    
    def fit_transform(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        تطبيق جميع خطوات المعالجة على بيانات التدريب
        """
        logger.info("=" * 60)
        logger.info("بدء خط أنابيب المعالجة المسبقة")
        logger.info("=" * 60)
        
        # فصل الميزات والهدف
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # تحديد أنواع الميزات
        self.identify_features(X, target_col)
        
        # فصل البيانات
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # ترميز الميزات الفئوية
        X_train, X_val, X_test = self.encode_features(
            X_train, X_val, X_test, self.categorical_features
        )
        
        # تطبيع الميزات الرقمية
        X_train, X_val, X_test = self.scale_features(
            X_train, X_val, X_test, self.numerical_features
        )
        
        logger.info("=" * 60)
        logger.info("✅ اكتملت المعالجة المسبقة")
        logger.info("=" * 60)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names
        }
    
    def save(self, path: str):
        """حفظ المعالج الموجود"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features
            }, f)
        logger.info(f"✅ تم حفظ المعالج في {path}")
    
    def load(self, path: str):
        """تحميل المعالج المحفوظ"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoders = data['label_encoders']
            self.feature_names = data['feature_names']
            self.categorical_features = data['categorical_features']
            self.numerical_features = data['numerical_features']
        logger.info(f"✅ تم تحميل المعالج من {path}")
