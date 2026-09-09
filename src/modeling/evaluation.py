# src/modeling/evaluation.py
"""
Advanced Model Evaluation - PR-AUC, Calibration Curves, etc.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AdvancedEvaluator:
    """
    تقييم متقدم للنموذج مع PR-AUC والمنحنيات المعايرة
    """
    
    def __init__(self):
        self.results = {}
        logger.info("تم تهيئة مقيّم النموذج المتقدم")
    
    def compute_pr_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        حساب Precision-Recall AUC (PR-AUC)
        """
        logger.info("📊 حساب PR-AUC...")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        avg_precision = average_precision_score(y_true, y_proba)
        
        logger.info(f"✅ PR-AUC: {pr_auc:.4f}")
        logger.info(f"✅ Average Precision: {avg_precision:.4f}")
        
        return {
            'pr_auc': pr_auc,
            'average_precision': avg_precision,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    
    def compute_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        حساب ROC-AUC
        """
        logger.info("📊 حساب ROC-AUC...")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        logger.info(f"✅ ROC-AUC: {roc_auc:.4f}")
        
        return {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def evaluate_at_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                             threshold: float = 0.5) -> Dict[str, float]:
        """
        تقييم النموذج عند عتبة معينة
        """
        y_pred = (y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'threshold': threshold,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Sensitivity/Recall
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
            'fdr': fp / (fp + tp) if (fp + tp) > 0 else 0,  # False Discovery Rate
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'f1': 2 * (metrics['precision'] * metrics['tpr']) / (metrics['precision'] + metrics['tpr'] + 1e-10)
                   if 'precision' in metrics else 0
        }
        
        return metrics
    
    def generate_detailed_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_proba: np.ndarray) -> Dict[str, Any]:
        """
        إنشاء تقرير تفصيلي للنموذج
        """
        logger.info("📋 إنشاء تقرير تفصيلي...")
        
        report = {
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc_metrics': self.compute_roc_auc(y_true, y_proba),
            'pr_metrics': self.compute_pr_auc(y_true, y_proba),
            'threshold_metrics': self.evaluate_at_threshold(y_true, y_proba, threshold=0.5)
        }
        
        logger.info("✅ اكتمل التقرير التفصيلي")
        return report
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                              n_bins: int = 10) -> Dict[str, Any]:
        """
        رسم منحنى المعايرة
        """
        logger.info("📈 رسم منحنى المعايرة...")
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_sums = np.zeros(n_bins)
        bin_total = np.zeros(n_bins)
        bin_accuracy = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i+1])
            if mask.sum() > 0:
                bin_sums[i] = y_proba[mask].mean()
                bin_total[i] = mask.sum()
                bin_accuracy[i] = y_true[mask].mean()
        
        # حساب ECE (Expected Calibration Error)
        ece = np.average(
            np.abs(bin_sums - bin_accuracy),
            weights=bin_total
        )
        
        logger.info(f"✅ ECE (Expected Calibration Error): {ece:.4f}")
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracy': bin_accuracy,
            'bin_confidence': bin_sums,
            'ece': ece,
            'bin_counts': bin_total
        }
