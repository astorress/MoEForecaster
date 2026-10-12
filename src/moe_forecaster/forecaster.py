"""
MoEForecaster: an ensemble of expert models combined by adaptive gating.

The class lives in a module (rather than inside notebook 08) so that the
serialized model (models/production/moe_forecaster.pkl) can be loaded from any
script or notebook with `pickle.load`, as long as `src/` is on the PYTHONPATH.
"""

from datetime import datetime

import numpy as np
import pandas as pd


class MoEForecaster:
    """
    Mixture-of-Experts forecaster for hourly electricity demand of Colombia's SIN.

    Adaptive gating:
    - High confidence (min_conf >= threshold): hard gating -> most likely expert
    - Low confidence (min_conf < threshold): soft gating -> probability-weighted
      combination of experts

    Parameters
    ----------
    experts : dict
        {expert_id: fitted model} where expert_id is "<day_cluster>_<week_cluster>"
    classifier_day : sklearn estimator with predict_proba (daily-profile cluster)
    classifier_week : sklearn estimator with predict_proba (weekly-profile cluster)
    label_encoder_day : LabelEncoder for type_day
    label_encoder_week : LabelEncoder for type_week
    feature_cols : list of str -- expert feature columns, in order
    clf_feature_cols : list of str -- classifier feature columns
    gating_threshold : float -- minimum confidence required for hard gating
    tau : float -- temperature for distance-based soft gating
    min_prob_soft : float -- minimum joint probability for an expert to take part in soft gating
    version : str -- model version (reported in the model card)
    """

    def __init__(self, experts, classifier_day, classifier_week,
                 label_encoder_day, label_encoder_week,
                 feature_cols, clf_feature_cols,
                 gating_threshold=0.75, tau=1.0, min_prob_soft=0.05,
                 version="1.0.0"):
        self.experts             = experts
        self.clf_day             = classifier_day
        self.clf_week            = classifier_week
        self.le_day              = label_encoder_day
        self.le_week             = label_encoder_week
        self.feature_cols        = feature_cols
        self.clf_feature_cols    = clf_feature_cols
        self.gating_threshold    = gating_threshold
        self.tau                 = tau
        self.min_prob_soft       = min_prob_soft
        self.version             = version
        self.created_at          = datetime.now().isoformat()

    def _get_expert_features(self, X):
        """X can be a DataFrame or a NumPy array."""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_cols].values
        return X

    def _get_clf_features(self, X):
        if isinstance(X, pd.DataFrame):
            available = [c for c in self.clf_feature_cols if c in X.columns]
            if len(available) < len(self.clf_feature_cols):
                return X[self.feature_cols].values  # fall back to the expert features
            return X[available].values
        return X  # already an ndarray: use as is

    def predict(self, X):
        """
        Predict with adaptive gating.

        X can be:
        - a pd.DataFrame with named columns
        - a np.ndarray with columns in the order of self.feature_cols

        Returns: np.ndarray of predictions (kWh)
        """
        X_exp = self._get_expert_features(X)
        X_clf_arr = self._get_clf_features(X)
        n = len(X_exp)
        predictions = np.zeros(n)

        # Cluster probabilities from both classifiers
        probs_day  = self.clf_day.predict_proba(X_clf_arr)   # (n, k_day)
        probs_week = self.clf_week.predict_proba(X_clf_arr)  # (n, k_week)
        classes_day  = self.le_day.classes_
        classes_week = self.le_week.classes_
        max_day  = probs_day.max(axis=1)
        max_week = probs_week.max(axis=1)
        min_conf = np.minimum(max_day, max_week)

        for i in range(n):
            x_i = X_exp[[i]]

            if min_conf[i] >= self.gating_threshold:
                # Hard gating: use the most likely expert
                cd = classes_day[np.argmax(probs_day[i])]
                cw = classes_week[np.argmax(probs_week[i])]
                eid = f'{cd}_{cw}'
                if eid in self.experts:
                    predictions[i] = self.experts[eid].predict(x_i)[0]
                else:
                    # Fallback: the requested expert does not exist
                    predictions[i] = self._fallback_predict(x_i, probs_day[i], probs_week[i],
                                                            classes_day, classes_week)
            else:
                # Soft gating: probability-weighted combination
                predictions[i] = self._soft_predict(x_i, probs_day[i], probs_week[i],
                                                    classes_day, classes_week)

        return predictions

    def predict_with_metadata(self, X):
        """Predict and return per-hour gating metadata (expert, confidences, mode) for monitoring."""
        X_exp = self._get_expert_features(X)
        X_clf_arr = self._get_clf_features(X)
        n = len(X_exp)
        predictions = np.zeros(n)
        metadata = []

        probs_day  = self.clf_day.predict_proba(X_clf_arr)
        probs_week = self.clf_week.predict_proba(X_clf_arr)
        classes_day  = self.le_day.classes_
        classes_week = self.le_week.classes_
        max_day  = probs_day.max(axis=1)
        max_week = probs_week.max(axis=1)
        min_conf = np.minimum(max_day, max_week)

        for i in range(n):
            x_i = X_exp[[i]]
            cd = classes_day[np.argmax(probs_day[i])]
            cw = classes_week[np.argmax(probs_week[i])]
            eid = f'{cd}_{cw}'

            if min_conf[i] >= self.gating_threshold:
                mode = 'hard'
                if eid in self.experts:
                    predictions[i] = self.experts[eid].predict(x_i)[0]
                else:
                    predictions[i] = self._fallback_predict(x_i, probs_day[i], probs_week[i],
                                                            classes_day, classes_week)
                    mode = 'hard_fallback'
            else:
                mode = 'soft'
                predictions[i] = self._soft_predict(x_i, probs_day[i], probs_week[i],
                                                    classes_day, classes_week)

            metadata.append({
                'expert_id': eid,
                'conf_day': round(float(max_day[i]), 4),
                'conf_week': round(float(max_week[i]), 4),
                'min_conf': round(float(min_conf[i]), 4),
                'gating_mode': mode,
            })

        return predictions, pd.DataFrame(metadata)

    def _soft_predict(self, x_i, probs_d, probs_w, classes_d, classes_w):
        pred, w_total = 0.0, 0.0
        for di, cd in enumerate(classes_d):
            for wi, cw in enumerate(classes_w):
                p = probs_d[di] * probs_w[wi]
                if p > self.min_prob_soft:
                    eid = f'{cd}_{cw}'
                    if eid in self.experts:
                        pred += p * self.experts[eid].predict(x_i)[0]
                        w_total += p
        if w_total > 0:
            return pred / w_total
        return self._fallback_predict(x_i, probs_d, probs_w, classes_d, classes_w)

    def _fallback_predict(self, x_i, probs_d, probs_w, classes_d, classes_w):
        """Last-resort prediction: average of all available experts."""
        if not self.experts:
            raise RuntimeError("MoEForecaster has no experts: cannot predict.")
        return np.mean([m.predict(x_i)[0] for m in self.experts.values()])

    def get_model_card(self):
        return {
            'version': self.version,
            'created_at': self.created_at,
            'n_experts': len(self.experts),
            'expert_ids': list(self.experts.keys()),
            'gating_threshold': self.gating_threshold,
            'tau': self.tau,
            'feature_count': len(self.feature_cols),
            'clf_feature_count': len(self.clf_feature_cols),
        }
