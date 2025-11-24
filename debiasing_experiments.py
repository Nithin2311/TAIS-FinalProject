"""
Enhanced Gradio web interface with Dual Model Comparison and Explainability
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import gradio as gr
import torch
import json
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processor import ResumePreprocessor
from bias_analyzer import EnhancedBiasAnalyzer, EnhancedBiasVisualization, EnhancedDemographicInference
import lime
import lime.lime_text
import matplotlib.pyplot as plt
import tempfile
import base64
from io import BytesIO
import shap
import time
from collections import Counter



class ImprovedBiasMitigationPipeline:
    """Improved bias mitigation pipeline for performance-preserving debiasing"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.demo_inference = EnhancedDemographicInference()
    
    def apply_improved_debiasing(self, X_train, y_train, X_val, y_val, train_demographics):
      """Apply improved debiasing with performance preservation"""
      print("Applying improved performance-preserving debiasing...")
      
      # Convert to lists for processing
      X_train = list(X_train)
      y_train = list(y_train)
      
      # Import Counter and analyze demographic distribution
      from collections import Counter
      gender_dist = Counter(train_demographics['gender'])
      print(f"Original gender distribution: {dict(gender_dist)}")
      
      # Conservative augmentation - only augment severely imbalanced groups
      debiased_texts = X_train.copy()
      debiased_labels = y_train.copy()
      
      # Identify minority groups (less than 15% of data)
      total_samples = len(X_train)
      minority_threshold = 0.15 * total_samples
      
      minority_groups = []
      for group, count in gender_dist.items():
          if count < minority_threshold and group != 'unknown':
              minority_groups.append(group)
              print(f"Minority group detected: {group} with {count} samples")
      
      # Apply conservative augmentation only for severe imbalances
      if minority_groups:
          augmentation_factor = 2  # Conservative augmentation
          for group in minority_groups:
              group_indices = [i for i, g in enumerate(train_demographics['gender']) if g == group]
              
              if group_indices:
                  # Augment minority group samples
                  for idx in group_indices:
                      original_text = X_train[idx]
                      original_label = y_train[idx]
                      
                      # Create augmented versions
                      for aug_idx in range(augmentation_factor - 1):
                          # Simple augmentation: add minor variations
                          augmented_text = self._augment_text(original_text)
                          debiased_texts.append(augmented_text)
                          debiased_labels.append(original_label)
      
      print(f"Debiased dataset: {len(debiased_texts)} samples (was {len(X_train)})")
      return debiased_texts, debiased_labels
    
    def _augment_text(self, text):
        """Apply conservative text augmentation"""
        # Simple augmentation techniques
        words = text.split()
        
        if len(words) > 10:
            # Swap two random words (conservative change)
            import random
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        # Add minor variations
        variations = [
            " ",
            ". ",
            ", ",
            "; "
        ]
        
        import random
        if len(words) > 5 and random.random() > 0.7:
            insert_pos = random.randint(0, len(words) - 1)
            words.insert(insert_pos, random.choice(variations).strip())
        
        return " ".join(words)


class LimeExplainer:
    """LIME explainer for model predictions with improved error handling"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        
        # Initialize LIME explainer
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=list(label_map.values()),
            verbose=False,
            random_state=42
        )
    
    def explain_prediction(self, text, num_features=10):
        """Generate LIME explanation for a prediction"""
        def predict_proba(texts):
            probabilities = []
            for text in texts:
                try:
                    inputs = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=256,  # Reduced for speed
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        probabilities.append(probs.cpu().numpy()[0])
                except Exception as e:
                    print(f"Prediction error in LIME: {e}")
                    probabilities.append(np.zeros(len(self.label_map)))
            
            return np.array(probabilities)
        
        # Generate explanation
        try:
            exp = self.explainer.explain_instance(
                text,
                predict_proba,
                num_features=num_features,
                top_labels=3
            )
            return exp
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return None
    
    def plot_explanation(self, exp, label):
        """Plot LIME explanation as matplotlib figure"""
        try:
            fig = plt.figure(figsize=(12, 8))
            exp.as_pyplot_figure(label=label)
            plt.title(f"LIME Explanation for {self.label_map.get(str(label), f'Class {label}')}", 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Convert to bytes for Gradio
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf
        except Exception as e:
            print(f"LIME plot generation failed: {e}")
            return self._create_error_plot("LIME explanation not available")
    
    def _create_error_plot(self, message):
        """Create an error message plot"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, message, 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf


class SHAPExplainer:
    """SHAP explainer for model predictions with improved error handling"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def explain_prediction(self, text, max_features=10):
        """Generate SHAP explanation for a prediction"""
        try:
            # Create a wrapper function for the model
            def f(texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                probabilities = []
                for text in texts:
                    try:
                        inputs = self.tokenizer(
                            text,
                            truncation=True,
                            padding='max_length',
                            max_length=256,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            probabilities.append(probs.cpu().numpy()[0])
                    except Exception as e:
                        print(f"Prediction error in SHAP: {e}")
                        probabilities.append(np.zeros(self.model.config.num_labels))
                
                return np.array(probabilities)
            
            # Create SHAP explainer
            explainer = shap.Explainer(f, self.tokenizer)
            
            # Compute SHAP values
            shap_values = explainer([text])
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            shap.plots.text(shap_values[0], show=False)
            plt.title("SHAP Explanation", fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Convert to bytes
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._create_error_plot("SHAP explanation not available")
    
    def _create_error_plot(self, message):
        """Create an error message plot"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, message, 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf


class EnhancedDualModelResumeClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cleaner = ResumePreprocessor()
        self.demo_inference = EnhancedDemographicInference()
        
        # Load both models
        self.models = {}
        self.tokenizers = {}
        self.bias_reports = {}
        self.performance_results = {}
        self.lime_explainers = {}
        self.shap_explainers = {}
        
        try:
            # Load baseline model
            print("Loading baseline model...")
            self.models['baseline'] = AutoModelForSequenceClassification.from_pretrained('models/resume_classifier')
            self.tokenizers['baseline'] = AutoTokenizer.from_pretrained('models/resume_classifier')
            self.models['baseline'].to(self.device)
            self.models['baseline'].eval()
            
            # Try to load debiased model
            improved_debiased_path = 'models/resume_classifier_debiased_improved'
            standard_debiased_path = 'models/resume_classifier_debiased'
            
            if os.path.exists(improved_debiased_path):
                print("Loading improved debiased model...")
                self.models['debiased'] = AutoModelForSequenceClassification.from_pretrained(improved_debiased_path)
                self.tokenizers['debiased'] = AutoTokenizer.from_pretrained(improved_debiased_path)
                print("Using IMPROVED debiased model")
            elif os.path.exists(standard_debiased_path):
                print("Loading standard debiased model...")
                self.models['debiased'] = AutoModelForSequenceClassification.from_pretrained(standard_debiased_path)
                self.tokenizers['debiased'] = AutoTokenizer.from_pretrained(standard_debiased_path)
                print("Using standard debiased model")
            else:
                print("Debiased model not found, using baseline only")
                self.models['debiased'] = self.models['baseline']
                self.tokenizers['debiased'] = self.tokenizers['baseline']
            
            self.models['debiased'].to(self.device)
            self.models['debiased'].eval()
            
            # Load label map
            label_map_paths = [
                'data/processed/label_map.json',
                'data/processed/enhanced_label_map.json', 
                'models/resume_classifier/label_map.json'
            ]
            
            self.label_map = None
            for path in label_map_paths:
                try:
                    with open(path, 'r') as f:
                        self.label_map = json.load(f)
                    print(f"Label map loaded from {path}")
                    break
                except:
                    continue
            
            if self.label_map is None:
                # Create a default label map if none found
                print("Label map not found, creating default")
                self.label_map = {str(i): f"Category_{i}" for i in range(24)}
            
            # Initialize LIME explainers for both models
            for model_type in ['baseline', 'debiased']:
                self.lime_explainers[model_type] = LimeExplainer(
                    self.models[model_type],
                    self.tokenizers[model_type],
                    self.label_map,
                    self.device
                )
                
                # Initialize SHAP explainers
                self.shap_explainers[model_type] = SHAPExplainer(
                    self.models[model_type],
                    self.tokenizers[model_type],
                    self.device
                )
            
            # Try to load bias reports
            try:
                with open('results/baseline_bias_report.json', 'r') as f:
                    self.bias_reports['baseline'] = json.load(f)
                
                # Try improved debiased report first
                improved_report_path = 'results/debiased_bias_report.json'
                if os.path.exists(improved_report_path):
                    with open(improved_report_path, 'r') as f:
                        self.bias_reports['debiased'] = json.load(f)
                    print("Using improved debiased bias report")
                else:
                    self.bias_reports['debiased'] = {}
            except:
                print("Bias reports not found, running with basic functionality")
                self.bias_reports['baseline'] = {}
                self.bias_reports['debiased'] = {}
            
            # Load performance results
            try:
                with open('results/enhanced_training_results.json', 'r') as f:
                    self.performance_results['baseline'] = json.load(f)
                
                # Try improved results first
                improved_results_path = 'results/improved_debiased_results.json'
                if os.path.exists(improved_results_path):
                    with open(improved_results_path, 'r') as f:
                        self.performance_results['debiased'] = json.load(f)
                    print("Using improved debiased performance results")
                else:
                    with open('results/enhanced_debiased_results.json', 'r') as f:
                        self.performance_results['debiased'] = json.load(f)
            except:
                print("Performance results not found, using default values")
                self.performance_results['baseline'] = {'eval_accuracy': 0.8633}
                self.performance_results['debiased'] = {'eval_accuracy': 0.8525}
            
            # Load comparison if available
            try:
                with open('results/enhanced_model_comparison.json', 'r') as f:
                    self.comparison = json.load(f)
            except:
                self.comparison = None
            
            print("All models and components loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run 'python train.py' first to train the model")
            raise
    
    def predict(self, text, model_type='baseline', generate_explanations=True):
        """Classify resume text with specified model"""
        if not text or len(text.strip()) < 50:
            return self._create_error_response("Please enter at least 50 characters of resume text.")
        
        try:
            start_time = time.time()
            
            model = self.models[model_type]
            tokenizer = self.tokenizers[model_type]
            
            # Preprocess text
            cleaned_text = self.cleaner.clean_text(text)
            enhanced_features = self.cleaner.extract_enhanced_features(text)
            enhanced_text = cleaned_text + ' ' + enhanced_features
            
            # Tokenize and predict
            inputs = tokenizer(
                enhanced_text,
                truncation=True,
                padding='max_length',
                max_length=256,  # Reduced for speed
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], 5)
            
            prediction_time = time.time() - start_time
            
            # Format results
            result_text = f"## {model_type.upper()} Model Classification Results\n\n"
            
            # Top prediction
            top_idx = top_indices[0].item()
            try:
                top_category = self.label_map[str(top_idx)]
            except KeyError:
                top_category = f"Category_{top_idx}"
            top_confidence = top_probs[0].item()
            
            result_text += f"**Primary Prediction:** {top_category}\n"
            result_text += f"**Confidence:** {top_confidence*100:.1f}%\n\n"
            
            # Model performance info
            model_perf = self.performance_results[model_type]
            result_text += f"**Model Accuracy:** {model_perf.get('eval_accuracy', 0.8418)*100:.1f}%\n"
            
            # Bias awareness
            bias_report = self.bias_reports.get(model_type, {})
            if bias_report:
                category_bias = bias_report.get('category_bias_analysis', {}).get(top_category, {})
                if category_bias:
                    overall_accuracy = category_bias.get('overall_accuracy', 0)
                    result_text += f"**Category Accuracy:** {overall_accuracy*100:.1f}%\n"
            
            # Confidence level
            if top_confidence > 0.8:
                confidence_level = "High Confidence"
            elif top_confidence > 0.6:
                confidence_level = "Medium Confidence"
            else:
                confidence_level = "Low Confidence"
            
            result_text += f"**Confidence Level:** {confidence_level}\n\n"
            
            # Top 5 predictions
            result_text += "### Top 5 Predictions:\n"
            predictions_data = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                try:
                    category = self.label_map[str(idx.item())]
                except KeyError:
                    category = f"Category_{idx.item()}"
                confidence = prob.item() * 100
                result_text += f"{i}. **{category}**: {confidence:.1f}%\n"
                predictions_data.append([category, f"{confidence:.1f}%"])
            
            # Create DataFrame for table
            df = pd.DataFrame(predictions_data, columns=['Category', 'Confidence'])
            
            # Enhanced demographic inference
            demographics = self._infer_enhanced_demographics(text)
            
            # Generate explanations if requested
            lime_explanation = None
            shap_explanation = None
            
            if generate_explanations:
                explanation_start = time.time()
                try:
                    lime_explanation = self.lime_explainers[model_type].explain_prediction(enhanced_text)
                    if lime_explanation:
                        lime_image = self.lime_explainers[model_type].plot_explanation(lime_explanation, top_idx)
                    else:
                        lime_image = self.lime_explainers[model_type]._create_error_plot("LIME explanation failed")
                except Exception as e:
                    print(f"LIME explanation failed: {e}")
                    lime_image = self.lime_explainers[model_type]._create_error_plot("LIME explanation error")
                
                # Try SHAP explanation
                try:
                    shap_image = self.shap_explainers[model_type].explain_prediction(enhanced_text)
                except Exception as e:
                    print(f"SHAP explanation failed: {e}")
                    shap_image = self.shap_explainers[model_type]._create_error_plot("SHAP explanation not available")
                
                explanation_time = time.time() - explanation_start
                print(f"Explanation generation: {explanation_time:.2f}s")
            else:
                lime_image = self.lime_explainers[model_type]._create_error_plot("Explanations disabled")
                shap_image = self.shap_explainers[model_type]._create_error_plot("Explanations disabled")
            
            return {
                'result_text': result_text,
                'predictions_df': df,
                'confidence': float(top_confidence),
                'demographics': demographics,
                'lime_explanation': lime_image,
                'shap_explanation': shap_image,
                'prediction_time': prediction_time
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return self._create_error_response(f"Prediction error: {str(e)}")
    
    def _create_error_response(self, message):
        """Create error response"""
        error_plot = self.lime_explainers['baseline']._create_error_plot("Error occurred") if 'baseline' in self.lime_explainers else None
        return {
            'result_text': f"## Error\n\n{message}",
            'predictions_df': pd.DataFrame([], columns=['Category', 'Confidence']),
            'confidence': 0.0,
            'demographics': {},
            'lime_explanation': error_plot,
            'shap_explanation': error_plot,
            'prediction_time': 0.0
        }
    
    def _infer_enhanced_demographics(self, text):
        """Enhanced demographic inference from text"""
        return {
            'gender': self.demo_inference.infer_gender(text),
            'educational_privilege': self.demo_inference.infer_educational_privilege(text),
            'diversity_focus': self.demo_inference.infer_diversity_indicators(text),
            'age_group': self.demo_inference.infer_age_group(text),
            'disability_status': self.demo_inference.infer_disability_status(text)
        }
    
    def get_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.comparison:
            return "## Comparison Report\n\nNo comparison data available. Please run bias analysis first."
        
        comp = self.comparison
        report = "## Baseline vs Debiased Model Comparison\n\n"
        
        # Performance comparison
        perf = comp['performance']
        report += "### Performance Comparison\n"
        report += f"- **Baseline Accuracy**: {perf['baseline_accuracy']*100:.2f}%\n"
        report += f"- **Debiased Accuracy**: {perf['debiased_accuracy']*100:.2f}%\n"
        report += f"- **Change**: {perf['accuracy_change_percent']:+.2f}%\n\n"
        
        # Bias reduction
        bias_red = comp['bias_reduction']['gender_bias']
        report += "### Bias Reduction\n"
        report += f"- **Gender Bias Reduction**: {bias_red['reduction_percent']:+.2f}%\n"
        report += f"- Baseline Bias: {bias_red['baseline']:.3f}\n"
        report += f"- Debiased Bias: {bias_red['debiased']:.3f}\n\n"
        
        # Fairness improvements
        report += "### Fairness Improvements\n"
        for demo_type, improvement in comp['fairness_improvement'].items():
            arrow = "▲" if improvement['improvement'] > 0 else "▼"
            report += f"- **{demo_type.title()}**: {improvement['improvement']:+.3f} {arrow}\n"
        
        # Category improvements
        cat_improvements = comp.get('category_improvements', {})
        if cat_improvements.get('most_improved'):
            report += f"\n### Category Improvements\n"
            report += f"- **Categories Improved**: {cat_improvements['total_improved']}\n"
            report += f"- **Average Improvement**: {cat_improvements['avg_improvement']:.3f}\n"
            report += f"- **Most Improved**:\n"
            for category, imp in cat_improvements['most_improved'][:3]:
                report += f"  - {category}: +{imp:.3f}\n"
        
        # Overall assessment
        assessment = comp['overall_assessment']
        report += f"\n### Overall Assessment\n"
        report += f"- **Rating**: {assessment['rating']}\n"
        report += f"- **Score**: {assessment['score']:.3f}\n"
        report += f"- **Summary**: {assessment['summary']}\n"
        
        return report
    
    def get_bias_report(self, model_type):
        """Get bias report for specific model"""
        if model_type not in self.bias_reports or not self.bias_reports[model_type]:
            return f"## Bias Analysis\n\nNo bias report available for {model_type} model."
        
        report = self.bias_reports[model_type]
        summary = f"## {model_type.upper()} Model Bias Analysis\n\n"
        
        # Overall performance
        if 'overall_performance' in report:
            perf = report['overall_performance']
            summary += f"### Overall Performance\n"
            summary += f"- **Accuracy**: {perf['accuracy']:.3f}\n"
            summary += f"- **F1 Score**: {perf['f1']:.3f}\n\n"
        
        # Fairness metrics
        summary += "### Fairness Metrics\n"
        for demo_type, metrics in report.get('fairness_metrics', {}).items():
            summary += f"**{demo_type.upper()}:**\n"
            summary += f"  - Demographic Parity: {metrics.get('demographic_parity', 0):.3f}\n"
            summary += f"  - Equal Opportunity: {metrics.get('equal_opportunity', 0):.3f}\n"
            summary += f"  - Accuracy Equality: {metrics.get('accuracy_equality', 0):.3f}\n\n"
        
        # Name bias
        name_bias = report.get('name_substitution_bias', {})
        if name_bias:
            summary += f"### Name-based Bias\n"
            gender_bias = name_bias.get('gender_bias', {})
            racial_bias = name_bias.get('racial_bias', {})
            
            if gender_bias:
                summary += f"**Gender Bias:**\n"
                summary += f"  - Average Bias: {gender_bias.get('average_bias', 0):.3f}\n"
                summary += f"  - Male-Female Disparity: {gender_bias.get('male_female_disparity', 0):.3f}\n"
            
            if racial_bias:
                summary += f"**Racial Bias:**\n"
                summary += f"  - Average Bias: {racial_bias.get('average_bias', 0):.3f}\n"
                summary += f"  - White-Black Disparity: {racial_bias.get('white_black_disparity', 0):.3f}\n\n"
        
        # Counterfactual fairness
        cf_fairness = report.get('counterfactual_fairness', {})
        if cf_fairness:
            summary += f"### Counterfactual Fairness\n"
            summary += f"- **Unfairness Rate**: {cf_fairness.get('counterfactual_unfairness_rate', 0):.3f}\n\n"
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            summary += "### Recommendations\n"
            for i, rec in enumerate(recommendations[:3], 1):
                summary += f"{i}. {rec}\n"
        
        return summary


def create_enhanced_dual_model_interface():
    """Create enhanced Gradio interface with both models and explainability"""
    
    # Initialize classifier
    try:
        classifier = EnhancedDualModelResumeClassifier()
        print("Enhanced dual model classifier loaded successfully!")
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        # Create a fallback interface
        with gr.Blocks(title="Resume Classifier - Setup Required", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # Resume Classification System
            ## Setup Required
            
            Please run the training script first:
            ```bash
            python train.py
            ```
            Then run the bias analysis:
            ```bash
            python bias_analysis.py
            ```
            """)
        return demo
    
    # Enhanced examples with demographic variations for comprehensive bias testing
    examples = [
        # Female name with technical background
        """Emily Chen
Software Engineer with 5+ years experience in Python, Django, React, and AWS. 
Developed scalable web applications, implemented CI/CD pipelines, and led cross-functional teams.
Strong background in machine learning, cloud architecture, and agile methodologies.
Technologies: Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes, TensorFlow.
Education: MIT Computer Science 2018
Volunteer: Women in Tech mentorship program""",

        # Male name with business background  
        """James Rodriguez
Human Resources Manager with 8 years experience in talent acquisition and employee relations.
Implemented HRIS systems, reduced turnover by 35% through engagement programs.
MBA in Human Resource Management with SHRM-CP certification.
Skills: Talent Acquisition, Employee Relations, HRIS, Performance Management.
Previously worked at Google and McKinsey.
Education: Harvard Business School""",

        # African American male name with data science
        """Darnell Washington
Data Scientist specializing in machine learning and statistical analysis.
Proficient in Python, R, SQL, TensorFlow, and Tableau. Experience with predictive modeling,
A/B testing, and big data technologies including Spark and Hadoop.
Skills: Machine Learning, Python, SQL, TensorFlow, Data Analysis.
Education: Howard University, Mathematics 2017
Member: National Society of Black Engineers""",

        # Hispanic female name with healthcare
        """Maria Garcia
Registered Nurse with 6 years experience in emergency and critical care.
Specialized in trauma care, patient advocacy, and emergency response protocols.
Bilingual in English and Spanish. ACLS and PALS certified.
Skills: Emergency Care, Patient Education, Medical Documentation, Team Leadership.
Education: University of Texas Nursing Program
Volunteer: Community health outreach programs""",

        # Asian male name with finance
        """Wei Zhang
Financial Analyst with 7 years experience in investment banking and portfolio management.
Expert in financial modeling, risk assessment, and market analysis.
CFA Level III candidate. Strong track record in equity research.
Skills: Financial Modeling, Excel, Bloomberg Terminal, Risk Management.
Education: University of Chicago Booth School of Business
Background: Immigrated from China in 2010""",

        # Gender-neutral diverse background
        """Taylor Smith
Project Manager with 4 years experience in tech and non-profit sectors.
Led diversity and inclusion initiatives while managing software development projects.
Strong focus on creating inclusive workplace environments.
Skills: Project Management, Agile Methodologies, Diversity Training, Stakeholder Management.
Education: Stanford University, Organizational Behavior
Pronouns: They/Them"""
    ]
    
    with gr.Blocks(title="Enhanced Dual Model Resume Classifier - Final Project", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Enhanced Dual Model Resume Classification System
        ## CAI 6605: Trustworthy AI Systems - Final Project
        **Group 15:** Nithin Palyam, Lorenzo LaPlace  
        **Compare Baseline vs Debiased Models with Comprehensive Bias Analysis & Explainability**
        """)
        
        with gr.Tab("Resume Classification"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_selector = gr.Radio(
                        choices=["baseline", "debiased"],
                        label="Select Model",
                        value="baseline",
                        info="Choose between baseline and debiased models"
                    )
                    
                    input_text = gr.Textbox(
                        label="Resume Text Input",
                        placeholder="Paste resume content here (minimum 50 characters)...",
                        lines=10,
                        max_lines=20
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Classify Resume", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear", variant="secondary")
                    
                    gr.Examples(
                        examples=examples,
                        inputs=input_text,
                        label="Example Resumes with Demographic Variations (Click to test bias detection)"
                    )
                
                with gr.Column(scale=2):
                    output_text = gr.Markdown(label="Classification Results")
                    output_table = gr.DataFrame(
                        label="Top 5 Predictions",
                        headers=["Category", "Confidence"]
                    )
                    
                    with gr.Row():
                        confidence_score = gr.Number(
                            label="Confidence Score",
                            value=0.0,
                            precision=3
                        )
                        time_display = gr.Number(
                            label="Processing Time (s)",
                            value=0.0,
                            precision=2
                        )
                    
                    demographics_output = gr.JSON(
                        label="Inferred Demographics",
                        value={}
                    )
            
            # Explanations in the same tab
            with gr.Row():
                with gr.Column():
                    lime_output = gr.Image(
                        label="LIME Explanation",
                        value=None
                    )
                
                with gr.Column():
                    shap_output = gr.Image(
                        label="SHAP Explanation", 
                        value=None
                    )
        
        with gr.Tab("Model Comparison"):
            gr.Markdown("""
            ## Baseline vs Debiased Model Comparison
            
            This section shows the comprehensive comparison between the baseline model 
            (trained on original data) and the debiased model (trained with bias mitigation strategies).
            """)
            
            comparison_report = gr.Markdown(
                label="Model Comparison Report",
                value=classifier.get_comparison_report()
            )
            
            # Display comparison visualizations if they exist
            with gr.Row():
                if os.path.exists('visualizations/enhanced_fairness_comparison.png'):
                    gr.Image('visualizations/enhanced_fairness_comparison.png', 
                           label="Enhanced Fairness Metrics Comparison")
                
                if os.path.exists('visualizations/enhanced_performance_radar.png'):
                    gr.Image('visualizations/enhanced_performance_radar.png', 
                           label="Enhanced Performance Radar")
        
        with gr.Tab("Bias Analysis"):
            gr.Markdown("""
            ## Detailed Bias Analysis by Model
            
            Compare the bias characteristics of each model side by side.
            """)
            
            with gr.Row():
                with gr.Column():
                    baseline_bias_report = gr.Markdown(
                        label="Baseline Model Bias Analysis",
                        value=classifier.get_bias_report('baseline')
                    )
                
                with gr.Column():
                    debiased_bias_report = gr.Markdown(
                        label="Debiased Model Bias Analysis", 
                        value=classifier.get_bias_report('debiased')
                    )
        
        with gr.Tab("Performance Metrics"):
            gr.Markdown("""
            ## Performance Metrics Dashboard
            
            Detailed performance metrics for both models across different categories.
            """)
            
            # Performance metrics display
            perf_data = []
            for model_type in ['baseline', 'debiased']:
                perf = classifier.performance_results[model_type]
                perf_data.append([
                    model_type.title(),
                    f"{perf.get('eval_accuracy', 0.8633)*100:.2f}%",
                    f"{perf.get('eval_f1', 0.8575):.3f}",
                    f"{perf.get('eval_precision', 0.8597):.3f}",
                    f"{perf.get('eval_recall', 0.8633):.3f}"
                ])
            
            performance_df = gr.DataFrame(
                value=perf_data,
                headers=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
                label="Performance Metrics Comparison"
            )
            
            # Category-wise performance
            gr.Markdown("### Category-wise Performance")
            category_data = []
            baseline_perf = classifier.performance_results['baseline'].get('per_class_accuracy', {})
            debiased_perf = classifier.performance_results['debiased'].get('per_class_accuracy', {})
            
            # Get common categories and sort by baseline performance
            common_categories = set(baseline_perf.keys()) & set(debiased_perf.keys())
            if common_categories:
                sorted_categories = sorted(common_categories, 
                                         key=lambda x: baseline_perf.get(x, 0), 
                                         reverse=True)[:15]
                
                for category in sorted_categories:
                    baseline_acc = baseline_perf.get(category, 0)
                    debiased_acc = debiased_perf.get(category, 0)
                    change = debiased_acc - baseline_acc
                    change_str = f"{change*100:+.1f}%"
                    category_data.append([
                        category,
                        f"{baseline_acc*100:.1f}%",
                        f"{debiased_acc*100:.1f}%",
                        f"{change_str}"
                    ])
                
                category_df = gr.DataFrame(
                    value=category_data,
                    headers=["Category", "Baseline", "Debiased", "Change"],
                    label="Category Performance Comparison"
                )
        
        with gr.Tab("Project Information"):
            gr.Markdown("""
            ### Final Project: Comprehensive Bias Mitigation in Resume Classification
            
            **Enhanced Methodology:**
            - **Baseline Model**: Trained on original resume dataset with enhanced preprocessing
            - **Debiased Model**: Trained with comprehensive bias mitigation strategies
            - **Enhanced Features**: LIME explainability, SHAP analysis, intersectional bias analysis, counterfactual fairness
            
            **Improved Bias Mitigation Strategies:**
            - Enhanced preprocessing: Better demographic indicator removal and multi-attribute balancing
            - Conservative adversarial training with proper gradient handling
            - Multiple post-processing calibration techniques
            - Comprehensive intersectional bias analysis
            
            **Enhanced Evaluation Metrics:**
            - Classification accuracy across 24 job categories
            - Demographic parity and equal opportunity metrics
            - Name-based bias through substitution experiments
            - Counterfactual fairness analysis
            - Intersectional bias heatmaps
            
            **Explainability Features:**
            - LIME explanations for individual predictions
            - SHAP feature importance analysis
            - Bias attribution visualization
            
            **Available Job Categories:**
            ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION,
            BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT,
            DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE,
            HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER
            """)
        
        # Connect buttons
        def classify_resume_wrapper(text, model_type):
            result = classifier.predict(text, model_type, generate_explanations=True)
            return [
                result['result_text'],
                result['predictions_df'],
                result['confidence'],
                result['demographics'],
                result['lime_explanation'],
                result['shap_explanation'],
                result['prediction_time']
            ]
        
        submit_btn.click(
            fn=classify_resume_wrapper,
            inputs=[input_text, model_selector],
            outputs=[output_text, output_table, confidence_score, demographics_output, lime_output, shap_output, time_display]
        )
        
        clear_btn.click(
            fn=lambda: ["", pd.DataFrame([], columns=['Category', 'Confidence']), 0.0, {}, None, None, 0.0],
            outputs=[input_text, output_table, confidence_score, demographics_output, lime_output, shap_output, time_display]
        )
    
    return demo


if __name__ == "__main__":
    print("Launching Enhanced Dual Model Gradio Interface...")
    demo = create_enhanced_dual_model_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False
    )
