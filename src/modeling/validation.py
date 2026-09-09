# src/modeling/validation.py
"""
Cross-Validation & Hyperparameter Tuning - التحقق من الصحة وضبط المعاملات
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import (
    cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
import time

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    التحقق المتقاطع من النموذج
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.cv_results = {}
        logger.info(f"تم تهيئة مدقق التحقق المتقاطع (n_splits={n_splits})")
    
    def perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        إجراء التحقق المتقاطع على النموذج
        """
        logger.info(f"إجراء التحقق المتقاطع ({self.n_splits}-fold)...")
        
        scorers = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc_weighted'
        }
        
        cv_results = cross_validate(
            model, X, y,
            cv=self.n_splits,
            scoring=scorers,
            return_train_score=True,
            n_jobs=-1
        )
        
        # حساب المتوسطات والانحرافات المعيارية
        summary = {}
        for metric in scorers.keys():
            test_scores = cv_results[f'test_{metric}']
            summary[f'{metric}_mean'] = np.mean(test_scores)
            summary[f'{metric}_std'] = np.std(test_scores)
            logger.info(f"  {metric}: {summary[f'{metric}_mean']:.4f} (+/- {summary[f'{metric}_std']:.4f})")
        
        self.cv_results = summary
        return summary


class HyperparameterTuner:
    """
    ضبط معاملات النموذج
    """
    
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.best_params = {}
        self.best_score = 0
        logger.info("تم تهيئة منظم المعاملات")
    
    def grid_search(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                   param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        بحث شامل عن أفضل المعاملات
        """
        logger.info("🔍 بدء البحث الشامل عن المعاملات...")
        
        start_time = time.time()
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            scoring='roc_auc_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        search_time = time.time() - start_time
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        logger.info(f"✅ البحث الشامل اكتمل في {search_time:.2f} ثانية")
        logger.info(f"⭐ أفضل درجة: {self.best_score:.4f}")
        logger.info(f"📋 أفضل المعاملات: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
    
    def random_search(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                     param_dist: Dict[str, List], n_iter: int = 20) -> Dict[str, Any]:
        """
        بحث عشوائي عن أفضل المعاملات (أسرع من البحث الشامل)
        """
        logger.info(f"🎲 بدء البحث العشوائي ({n_iter} تكرار)...")
        
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            model,
            param_dist,
            n_iter=n_iter,
            cv=self.cv_folds,
            scoring='roc_auc_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        search_time = time.time() - start_time
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        logger.info(f"✅ البحث العشوائي اكتمل في {search_time:.2f} ثانية")
        logger.info(f"⭐ أفضل درجة: {self.best_score:.4f}")
        logger.info(f"📋 أفضل المعاملات: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': pd.DataFrame(random_search.cv_results_)
        }
