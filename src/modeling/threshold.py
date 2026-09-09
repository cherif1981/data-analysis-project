# src/modeling/threshold.py
"""
Threshold Optimization & Calibration - تحسين العتبات ومعايرة النموذج
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    تحسين عتبة التصنيف بناءً على مقاييس مختلفة
    """
    
    def __init__(self):
        self.optimal_threshold = 0.5
        self.threshold_metrics = {}
        logger.info("تم تهيئة محسّن العتبات")
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
        """
        إيجاد أفضل عتبة بناءً على مقياس محدد
        
        Args:
            y_true: القيم الفعلية
            y_proba: الاحتمالات المتنبأ بها
            metric: المقياس المستخدم ('f1', 'precision', 'recall', 'balanced')
        """
        logger.info(f"🎯 البحث عن أفضل عتبة بناءً على {metric}...")
        
        thresholds = np.arange(0.0, 1.01, 0.01)
        best_score = 0
        best_threshold = 0.5
        threshold_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced':
                # Balance بين precision و recall
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                score = (precision + recall) / 2
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            threshold_scores.append({
                'threshold': threshold,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        self.threshold_metrics = pd.DataFrame(threshold_scores)
        
        logger.info(f"✅ أفضل عتبة: {best_threshold:.3f}")
        logger.info(f"   أفضل درجة ({metric}): {best_score:.4f}")
        
        return best_threshold, {
            'threshold': best_threshold,
            'score': best_score,
            'metric': metric
        }
    
    def find_roc_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        إيجاد العتبة المثلى بناءً على منحنى ROC (Youden's Index)
        """
        logger.info("📊 البحث عن أفضل عتبة من منحنى ROC...")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Youden's Index = TPR - FPR
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"✅ أفضل عتبة (ROC): {optimal_threshold:.3f}")
        logger.info(f"   TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
        
        return optimal_threshold, {
            'threshold': optimal_threshold,
            'tpr': float(tpr[optimal_idx]),
            'fpr': float(fpr[optimal_idx]),
            'youden_index': float(youden_index[optimal_idx])
        }
    
    def find_pr_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        إيجاد العتبة المثلى بناءً على منحنى Precision-Recall
        """
        logger.info("📈 البحث عن أفضل عتبة من منحنى PR...")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # F1 Score على منحنى PR
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"✅ أفضل عتبة (PR): {optimal_threshold:.3f}")
        logger.info(f"   Precision: {precision[optimal_idx]:.4f}, Recall: {recall[optimal_idx]:.4f}")
        
        return optimal_threshold, {
            'threshold': optimal_threshold,
            'precision': float(precision[optimal_idx]),
            'recall': float(recall[optimal_idx]),
            'f1': float(f1_scores[optimal_idx])
        }


class ModelCalibrator:
    """
    معايرة النموذج لتحسين تقدير الاحتمالات
    """
    
    def __init__(self, method: str = 'sigmoid'):
        """
        Args:
            method: 'sigmoid' أو 'isotonic'
        """
        self.method = method
        self.calibrated_model = None
        logger.info(f"تم تهيئة معايِر النموذج (method={method})")
    
    def calibrate(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        معايرة النموذج باستخدام بيانات التحقق
        """
        logger.info(f"🔧 معايرة النموذج باستخدام طريقة {self.method}...")
        
        # تقسيم بيانات التدريب للمعايرة
        self.calibrated_model = CalibratedClassifierCV(
            model,
            method=self.method,
            cv='prefit'  # استخدام نموذج مدرب مسبقاً
        )
        
        # معايرة على بيانات التحقق
        self.calibrated_model.fit(X_val, y_val)
        
        logger.info("✅ اكتملت معايرة النموذج")
        return self.calibrated_model
    
    def compare_calibration(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray,
                           y_proba: np.ndarray) -> Dict[str, Any]:
        """
        مقارنة الاحتمالات قبل وبعد المعايرة
        """
        logger.info("📊 مقارنة الاحتمالات قبل وبعد المعايرة...")
        
        # الاحتمالات المعايرة
        y_proba_calibrated = self.calibrated_model.predict_proba(X_cal)[:, 1]
        
        # حساب معايرة الخطأ (Expected Calibration Error)
        def compute_ece(y_true, y_proba, n_bins=10):
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_sums = np.zeros(n_bins)
            bin_total = np.zeros(n_bins)
            
            for i in range(n_bins):
                mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_sums[i] = np.abs(y_proba[mask].mean() - y_true[mask].mean())
                    bin_total[i] = mask.sum()
            
            return np.average(bin_sums, weights=bin_total)
        
        ece_before = compute_ece(y_cal, y_proba)
        ece_after = compute_ece(y_cal, y_proba_calibrated)
        
        logger.info(f"  Before: ECE = {ece_before:.4f}")
        logger.info(f"  After:  ECE = {ece_after:.4f}")
        logger.info(f"  Improvement: {(ece_before - ece_after) / ece_before * 100:.2f}%")
        
        return {
            'ece_before': ece_before,
            'ece_after': ece_after,
            'improvement': (ece_before - ece_after) / ece_before * 100
        }
