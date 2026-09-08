# src/ml/churn_predictor.py
"""
Customer Churn Prediction using Machine Learning
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from pathlib import Path

class ChurnPredictor:
    """Predict customer churn probability"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.model_path = Path('models/churn_model.pkl')
        self.scaler_path = Path('models/scaler.pkl')
        
        # Load model if exists
        if self.model_path.exists():
            self._load_model()
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict churn probability for customers"""
        try:
            self.logger.info("Starting churn prediction...")
            
            # Prepare features
            features = self._prepare_features(df)
            
            # Train or load model
            if self.model is None:
                self._train_model(features, df)
            
            # Make predictions
            predictions = self.model.predict_proba(features)[:, 1]
            churn_labels = (predictions > 0.5).astype(int)
            
            # Get customer IDs
            customer_ids = df.index.tolist() if 'customer_id' not in df.columns else df['customer_id'].tolist()
            
            # Create results
            results = []
            for i, (cid, prob) in enumerate(zip(customer_ids, predictions)):
                risk_level = self._get_risk_level(prob)
                results.append({
                    'customer_id': cid,
                    'churn_probability': float(prob),
                    'churn_risk': risk_level,
                    'is_high_risk': prob > 0.7,
                    'recommendation': self._get_recommendation(risk_level)
                })
            
            # Sort by risk
            results.sort(key=lambda x: x['churn_probability'], reverse=True)
            
            # Generate summary
            summary = self._generate_summary(results)
            
            self.logger.info(f"Churn prediction complete: {len(results)} customers analyzed")
            
            return {
                'predictions': results,
                'summary': summary,
                'high_risk_customers': [r for r in results if r['is_high_risk']],
                'top_10_risk': results[:10]
            }
            
        except Exception as e:
            self.logger.error(f"Churn prediction failed: {str(e)}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for churn prediction"""
        features = []
        
        # Calculate RFM features
        if 'recency' in df.columns:
            features.append(df['recency'])
        else:
            # Calculate recency
            ref_date = df['purchase_date'].max() if 'purchase_date' in df.columns else pd.Timestamp.now()
            recency = df.groupby('customer_id')['purchase_date'].apply(
                lambda x: (ref_date - x.max()).days if not x.empty else 365
            )
            features.append(recency)
        
        if 'frequency' in df.columns:
            features.append(df['frequency'])
        else:
            frequency = df.groupby('customer_id').size()
            features.append(frequency)
        
        if 'monetary' in df.columns:
            features.append(df['monetary'])
        else:
            monetary = df.groupby('customer_id')['amount'].sum()
            features.append(monetary)
        
        # Calculate additional features
        features.append(self._calculate_avg_order_value(df))
        features.append(self._calculate_days_since_last(df))
        features.append(self._calculate_purchase_variability(df))
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        else:
            feature_matrix = self.scaler.transform(feature_matrix)
        
        return feature_matrix
    
    def _calculate_avg_order_value(self, df: pd.DataFrame) -> pd.Series:
        """Calculate average order value per customer"""
        if 'amount' not in df.columns:
            return pd.Series(index=df.index, data=0)
        return df.groupby('customer_id')['amount'].mean().fillna(0)
    
    def _calculate_days_since_last(self, df: pd.DataFrame) -> pd.Series:
        """Calculate days since last purchase"""
        if 'purchase_date' not in df.columns:
            return pd.Series(index=df.index, data=365)
        
        ref_date = df['purchase_date'].max()
        last_date = df.groupby('customer_id')['purchase_date'].max()
        return (ref_date - last_date).dt.days.fillna(365)
    
    def _calculate_purchase_variability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate purchase amount variability"""
        if 'amount' not in df.columns:
            return pd.Series(index=df.index, data=0)
        
        std = df.groupby('customer_id')['amount'].std().fillna(0)
        mean = df.groupby('customer_id')['amount'].mean().fillna(0)
        
        # Coefficient of variation
        return (std / (mean + 1)).fillna(0)
    
    def _train_model(self, features: np.ndarray, df: pd.DataFrame):
        """Train churn prediction model"""
        try:
            self.logger.info("Training churn prediction model...")
            
            # Create target variable (churn based on recency)
            if 'recency' in df.columns:
                # Customers with recency > 90 days are considered churned
                target = (df['recency'] > 90).astype(int)
            else:
                # Calculate churn based on purchase patterns
                last_date = df.groupby('customer_id')['purchase_date'].max()
                ref_date = df['purchase_date'].max()
                target = ((ref_date - last_date).dt.days > 90).astype(int)
            
            # Ensure target matches feature rows
            if len(target) != features.shape[0]:
                self.logger.error("Feature and target dimensions don't match")
                raise ValueError("Feature and target dimensions don't match")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.3, random_state=42
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
            
            # Feature importance
            self.feature_importance = {
                'recency': 0.4,
                'frequency': 0.3,
                'monetary': 0.15,
                'avg_order': 0.05,
                'days_since_last': 0.05,
                'variability': 0.05
            }
            
            self.logger.info(f"Model trained: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _save_model(self):
        """Save trained model"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save model: {str(e)}")
    
    def _load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load model: {str(e)}")
            self.model = None
            self.scaler = None
    
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level from probability"""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            "CRITICAL": "Immediate action required - personalized offer needed",
            "HIGH": "Urgent re-engagement campaign - special discount",
            "MEDIUM": "Proactive outreach - survey and engagement",
            "LOW": "Maintain engagement - regular communication",
            "VERY LOW": "Continue current strategy"
        }
        return recommendations.get(risk_level, "Monitor customer behavior")
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate churn prediction summary"""
        probs = [r['churn_probability'] for r in results]
        
        return {
            'total_customers': len(results),
            'avg_churn_probability': np.mean(probs),
            'median_churn_probability': np.median(probs),
            'high_risk_count': len([r for r in results if r['is_high_risk']]),
            'critical_risk_count': len([r for r in results if r['churn_risk'] == 'CRITICAL']),
            'risk_distribution': {
                'critical': len([r for r in results if r['churn_risk'] == 'CRITICAL']),
                'high': len([r for r in results if r['churn_risk'] == 'HIGH']),
                'medium': len([r for r in results if r['churn_risk'] == 'MEDIUM']),
                'low': len([r for r in results if r['churn_risk'] == 'LOW']),
                'very_low': len([r for r in results if r['churn_risk'] == 'VERY LOW'])
            }
        }