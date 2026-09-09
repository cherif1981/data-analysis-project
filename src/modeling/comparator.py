# src/modeling/comparator.py
"""
Model Comparison - مقارنة النماذج
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    مقارنة عدة نماذج تعلم آلي
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.results = {}
        logger.info("تم تهيئة مقارن النماذج")
    
    def _initialize_models(self) -> Dict[str, Any]:
        """تهيئة مجموعة النماذج"""
        return {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        تدريب وتقييم جميع النماذج
        """
        logger.info("=" * 60)
        logger.info("بدء مقارنة النماذج")
        logger.info("=" * 60)
        
        comparison_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\n🔄 تدريب {model_name}...")
            
            # التدريب
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # التنبؤ
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            y_proba_val = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred_val
            y_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
            
            # التقييم
            val_metrics = self._calculate_metrics(y_val, y_pred_val, y_proba_val)
            test_metrics = self._calculate_metrics(y_test, y_pred_test, y_proba_test)
            
            comparison_results[model_name] = {
                'validation': val_metrics,
                'test': test_metrics,
                'train_time': train_time,
                'model': model
            }
            
            logger.info(f"✅ {model_name} - Validation Acc: {val_metrics['accuracy']:.4f}, "
                       f"ROC-AUC: {val_metrics['roc_auc']:.4f}")
            logger.info(f"   Test Acc: {test_metrics['accuracy']:.4f}, "
                       f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        
        self.results = comparison_results
        return comparison_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_proba: np.ndarray) -> Dict[str, float]:
        """حساب مقاييس الأداء"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba, average='weighted')
        }
    
    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """الحصول على أفضل نموذج بناءً على ROC-AUC"""
        best_model = None
        best_score = 0
        best_name = None
        
        for model_name, result in self.results.items():
            score = result['test']['roc_auc']
            if score > best_score:
                best_score = score
                best_name = model_name
                best_model = result
        
        logger.info(f"\n⭐ أفضل نموذج: {best_name} (ROC-AUC: {best_score:.4f})")
        return best_name, best_model
    
    def get_comparison_table(self) -> pd.DataFrame:
        """الحصول على جدول مقارنة النماذج"""
        data = []
        for model_name, result in self.results.items():
            data.append({
                'Model': model_name,
                'Val_Accuracy': result['validation']['accuracy'],
                'Val_ROC_AUC': result['validation']['roc_auc'],
                'Test_Accuracy': result['test']['accuracy'],
                'Test_ROC_AUC': result['test']['roc_auc'],
                'Train_Time': result['train_time']
            })
        
        return pd.DataFrame(data).sort_values('Test_ROC_AUC', ascending=False)
