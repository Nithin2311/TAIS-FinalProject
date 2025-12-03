"""
Fairness metrics for multi-class classification.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


class FairnessMetrics:
    """Fairness metrics for multi-class classification"""
    
    @staticmethod
    def equalized_odds_difference(y_true, y_pred, protected_attr, num_classes):
        """
        Equalized Odds: Equal TPR and FPR across groups
        Returns maximum difference in TPR and FPR across groups
        """
        protected_attr = np.array(protected_attr)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        group_metrics = {}
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                tpr_list = []
                fpr_list = []
                
                for cls in range(num_classes):
                    tp = np.sum((y_true_group == cls) & (y_pred_group == cls))
                    fn = np.sum((y_true_group == cls) & (y_pred_group != cls))
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tpr_list.append(float(tpr))
                    
                    fp = np.sum((y_true_group != cls) & (y_pred_group == cls))
                    tn = np.sum((y_true_group != cls) & (y_pred_group != cls))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fpr_list.append(float(fpr))
                
                group_metrics[group] = {
                    'tpr': float(np.mean(tpr_list)),
                    'fpr': float(np.mean(fpr_list))
                }
        
        tpr_values = [metrics['tpr'] for metrics in group_metrics.values()]
        fpr_values = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        return {
            'equalized_odds_difference': float(max(tpr_diff, fpr_diff)),
            'tpr_difference': float(tpr_diff),
            'fpr_difference': float(fpr_diff)
        }
    
    @staticmethod
    def treatment_equality_ratio(y_true, y_pred, protected_attr):
        """
        Treatment Equality: Ratio of false positive to false negative rates
        Should be equal across groups
        """
        protected_attr = np.array(protected_attr)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        group_ratios = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                fp = np.sum((y_true_group != y_pred_group) & (y_pred_group != -1))
                fn = np.sum((y_true_group != y_pred_group) & (y_pred_group == -1))
                
                ratio = float(fp / fn) if fn > 0 else float('inf')
                group_ratios.append(ratio)
        
        valid_ratios = [r for r in group_ratios if r != float('inf')]
        if len(valid_ratios) >= 2:
            max_ratio = max(valid_ratios)
            min_ratio = min(valid_ratios)
            if max_ratio > 0:
                normalized_diff = (max_ratio - min_ratio) / max_ratio
            else:
                normalized_diff = 0
        else:
            normalized_diff = 0
        
        return float(normalized_diff)
    
    @staticmethod
    def conditional_demographic_parity(y_true, y_pred, protected_attr, legitimate_factors):
        """
        Conditional Demographic Parity: Fairness given legitimate factors
        legitimate_factors: list of arrays for each legitimate factor
        """
        protected_attr = np.array(protected_attr)
        y_pred = np.array(y_pred)
        
        factor_combinations = set(zip(*legitimate_factors))
        
        disparities = []
        
        for combo in factor_combinations:
            combo_mask = np.ones(len(y_pred), dtype=bool)
            for i, factor in enumerate(legitimate_factors):
                combo_mask &= (np.array(factor) == combo[i])
            
            if np.sum(combo_mask) > 0:
                protected_subset = protected_attr[combo_mask]
                pred_subset = y_pred[combo_mask]
                
                groups = np.unique(protected_subset)
                group_rates = []
                
                for group in groups:
                    group_mask = protected_subset == group
                    if np.sum(group_mask) > 0:
                        positive_rate = float(np.mean(pred_subset[group_mask]))
                        group_rates.append(positive_rate)
                
                if len(group_rates) >= 2:
                    disparity = max(group_rates) - min(group_rates)
                    disparities.append(float(disparity))
        
        return float(np.mean(disparities)) if disparities else 0
    
    @staticmethod
    def intersectional_fairness(y_true, y_pred, protected_attrs_dict):
        """
        Intersectional fairness across multiple protected attributes
        protected_attrs_dict: dictionary of protected attributes
        """
        protected_names = list(protected_attrs_dict.keys())
        protected_values = list(protected_attrs_dict.values())
        
        intersectional_groups = set(zip(*protected_values))
        
        group_accuracies = {}
        
        for combo in intersectional_groups:
            group_mask = np.ones(len(y_true), dtype=bool)
            for i, values in enumerate(protected_values):
                group_mask &= (np.array(values) == combo[i])
            
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                accuracy = float(np.mean(y_true_group == y_pred_group))
                group_name = "_".join([f"{protected_names[i]}_{combo[i]}" 
                                      for i in range(len(combo))])
                group_accuracies[group_name] = {
                    'accuracy': accuracy,
                    'size': int(np.sum(group_mask))
                }
        
        if group_accuracies:
            accuracies = [info['accuracy'] for info in group_accuracies.values()]
            fairness_score = 1 - (max(accuracies) - min(accuracies))
        else:
            fairness_score = 1
        
        result = {
            'intersectional_fairness': float(fairness_score),
            'group_accuracies': group_accuracies
        }
        
        if group_accuracies:
            accuracies = [info['accuracy'] for info in group_accuracies.values()]
            result['min_accuracy'] = float(min(accuracies))
            result['max_accuracy'] = float(max(accuracies))
            result['accuracy_range'] = float(max(accuracies) - min(accuracies))
        else:
            result['min_accuracy'] = 0.0
            result['max_accuracy'] = 0.0
            result['accuracy_range'] = 0.0
        
        return result
    
    @staticmethod
    def comprehensive_fairness_report(y_true, y_pred, protected_attrs, num_classes):
        """Generate comprehensive fairness report"""
        report = {}
        
        report['demographic_parity'] = FairnessMetrics._demographic_parity(y_pred, protected_attrs)
        report['equalized_odds'] = FairnessMetrics.equalized_odds_difference(
            y_true, y_pred, protected_attrs, num_classes
        )
        
        unique_groups = np.unique(protected_attrs)
        group_accuracies = {}
        
        for group in unique_groups:
            group_mask = protected_attrs == group
            if np.sum(group_mask) > 0:
                accuracy = float(np.mean(y_true[group_mask] == y_pred[group_mask]))
                group_accuracies[str(group)] = accuracy
        
        report['group_accuracies'] = group_accuracies
        report['accuracy_equality'] = float(max(group_accuracies.values()) - min(group_accuracies.values())) if group_accuracies else 0
        
        return report
    
    @staticmethod
    def _demographic_parity(y_pred, protected_attr):
        """Helper method for demographic parity"""
        protected_attr = np.array(protected_attr)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        group_rates = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                positive_rate = float(np.mean(y_pred[group_mask]))
                group_rates.append(positive_rate)
        
        return float(max(group_rates) - min(group_rates)) if len(group_rates) >= 2 else 0
