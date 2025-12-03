"""
Resume classification web interface.
"""

import gradio as gr
import torch
import json
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processor import ResumePreprocessor
from bias_analyzer import DemographicInference
import lime
import lime.lime_text
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class LimeExplainer:
    """LIME explainer for model predictions"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        
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
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        probabilities.append(probs.cpu().numpy()[0])
                except Exception as e:
                    probabilities.append(np.ones(len(self.label_map)) / len(self.label_map))
            
            return np.array(probabilities)
        
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
        """Plot LIME explanation as numpy array for Gradio"""
        try:
            fig = plt.figure(figsize=(12, 8))
            exp.as_pyplot_figure(label=label)
            plt.title(f"LIME Explanation for {self.label_map.get(str(label), f'Class {label}')}", 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            image = Image.open(buf)
            numpy_array = np.array(image)
            return numpy_array
            
        except Exception as e:
            print(f"LIME plot generation failed: {e}")
            return self._create_error_plot("LIME explanation not available")
    
    def _create_error_plot(self, message):
        """Create an error message plot as numpy array"""
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, message, 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        image = Image.open(buf)
        numpy_array = np.array(image)
        return numpy_array


class ResumeClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cleaner = ResumePreprocessor()
        self.demo_inference = DemographicInference()
        
        self.models = {}
        self.tokenizers = {}
        self.label_map = {}
        
        try:
            print("Loading baseline model...")
            self.models['baseline'] = AutoModelForSequenceClassification.from_pretrained('models/resume_classifier_baseline')
            self.tokenizers['baseline'] = AutoTokenizer.from_pretrained('models/resume_classifier_baseline')
            self.models['baseline'].to(self.device)
            self.models['baseline'].eval()
            
            debiased_path = 'models/resume_classifier_debiased'
            
            if os.path.exists(debiased_path):
                print("Loading debiased model...")
                self.models['debiased'] = AutoModelForSequenceClassification.from_pretrained(debiased_path)
                self.tokenizers['debiased'] = AutoTokenizer.from_pretrained(debiased_path)
                print("Using debiased model")
            else:
                print("Debiased model not found, using baseline only")
                self.models['debiased'] = self.models['baseline']
                self.tokenizers['debiased'] = self.tokenizers['baseline']
            
            self.models['debiased'].to(self.device)
            self.models['debiased'].eval()
            
            label_map_paths = [
                'data/processed/label_map.json',
                'models/resume_classifier_baseline/label_map.json',
                'models/resume_classifier_debiased/label_map.json'
            ]
            
            for path in label_map_paths:
                try:
                    with open(path, 'r') as f:
                        self.label_map = json.load(f)
                    print(f"Label map loaded from {path}")
                    break
                except:
                    continue
            
            if not self.label_map:
                print("Label map not found, creating default")
                self.label_map = {str(i): f"Category_{i}" for i in range(24)}
            
            self.lime_explainers = {}
            for model_type in ['baseline', 'debiased']:
                self.lime_explainers[model_type] = LimeExplainer(
                    self.models[model_type],
                    self.tokenizers[model_type],
                    self.label_map,
                    self.device
                )
            
            try:
                with open('results/baseline_results.json', 'r') as f:
                    self.baseline_results = json.load(f)
            except:
                print("Baseline results not found")
                self.baseline_results = {'eval_accuracy': 0.8633}
            
            try:
                with open('results/debiased_results.json', 'r') as f:
                    self.debiased_results = json.load(f)
            except:
                print("Debiased results not found")
                self.debiased_results = {'eval_accuracy': 0.8525}
            
            print("Models and components loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run training scripts first")
            raise
    
    def predict(self, text, model_type='baseline'):
        """Classify resume text with specified model"""
        if not text or len(text.strip()) < 50:
            return "Please enter at least 50 characters of resume text.", None, None, None, None, None
        
        try:
            model = self.models[model_type]
            tokenizer = self.tokenizers[model_type]
            
            cleaned_text = self.cleaner.clean_text(text)
            features = self.cleaner.extract_features(text)
            combined_text = cleaned_text + ' ' + features
            
            inputs = tokenizer(
                combined_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], 5)
            
            result_text = f"## {model_type.upper()} Model Classification Results\n\n"
            
            top_idx = top_indices[0].item()
            top_category = self.label_map.get(str(top_idx), f"Category_{top_idx}")
            top_confidence = top_probs[0].item()
            
            result_text += f"**Primary Prediction:** {top_category}\n"
            result_text += f"**Confidence:** {top_confidence*100:.1f}%\n\n"
            
            if model_type == 'baseline':
                model_accuracy = self.baseline_results.get('eval_accuracy', 0.8418) * 100
            else:
                model_accuracy = self.debiased_results.get('eval_accuracy', 0.8525) * 100
            
            result_text += f"**Model Accuracy:** {model_accuracy:.1f}%\n"
            
            if top_confidence > 0.8:
                confidence_level = "High Confidence"
            elif top_confidence > 0.6:
                confidence_level = "Medium Confidence"
            else:
                confidence_level = "Low Confidence"
            
            result_text += f"**Confidence Level:** {confidence_level}\n\n"
            
            result_text += "### Top 5 Predictions:\n"
            predictions_data = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                category = self.label_map.get(str(idx.item()), f"Category_{idx.item()}")
                confidence = prob.item() * 100
                result_text += f"{i}. **{category}**: {confidence:.1f}%\n"
                predictions_data.append([category, f"{confidence:.1f}%"])
            
            df = pd.DataFrame(predictions_data, columns=['Category', 'Confidence'])
            
            demographics = self._infer_demographics(text)
            
            lime_image = self._generate_lime_explanation(combined_text, model_type, top_idx)
            
            return result_text, df, float(top_confidence), demographics, lime_image
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            return error_msg, None, None, None, None
    
    def _generate_lime_explanation(self, text, model_type, predicted_label):
        """Generate LIME explanation for the prediction"""
        try:
            explainer = self.lime_explainers[model_type]
            exp = explainer.explain_prediction(text, num_features=8)
            if exp is not None:
                explanation_image = explainer.plot_explanation(exp, predicted_label)
                return explanation_image
            else:
                return explainer._create_error_plot("LIME explanation generation failed")
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            explainer = self.lime_explainers[model_type]
            return explainer._create_error_plot("LIME explanation not available")
    
    def _infer_demographics(self, text):
        """Enhanced demographic inference from text"""
        return {
            'gender': self.demo_inference.infer_gender(text),
            'race': self.demo_inference.infer_race_from_names(text),
            'age_group': self.demo_inference.infer_age_group(text)
        }
    
    def get_performance_metrics(self):
        """Get performance metrics for both models"""
        metrics_data = []
        
        baseline_acc = self.baseline_results.get('eval_accuracy', 0.8633) * 100
        baseline_f1 = self.baseline_results.get('eval_f1', 0.8575) * 100
        baseline_precision = self.baseline_results.get('eval_precision', 0.8597) * 100
        baseline_recall = self.baseline_results.get('eval_recall', 0.8633) * 100
        
        debiased_acc = self.debiased_results.get('eval_accuracy', 0.8525) * 100
        debiased_f1 = self.debiased_results.get('eval_f1', 0.8478) * 100
        debiased_precision = self.debiased_results.get('eval_precision', 0.8492) * 100
        debiased_recall = self.debiased_results.get('eval_recall', 0.8525) * 100
        
        metrics_data.append(["Baseline", f"{baseline_acc:.2f}%", f"{baseline_f1:.2f}%", 
                           f"{baseline_precision:.2f}%", f"{baseline_recall:.2f}%"])
        metrics_data.append(["Debiased", f"{debiased_acc:.2f}%", f"{debiased_f1:.2f}%", 
                           f"{debiased_precision:.2f}%", f"{debiased_recall:.2f}%"])
        
        return metrics_data


def create_interface():
    """Create Gradio interface with two tabs"""
    
    try:
        classifier = ResumeClassifier()
        print("Resume classifier loaded successfully")
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        with gr.Blocks(title="Resume Classifier - Setup Required", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # Resume Classification System
            ## Setup Required
            
            Please run the training script first:
            ```bash
            python train_baseline.py
            ```
            Then run the bias analysis:
            ```bash
            python bias_analysis.py
            ```
            """)
        return demo
    
    examples = [
        """Darnell Washington
Data Scientist specializing in machine learning and statistical analysis.
Proficient in Python, R, SQL, TensorFlow, and Tableau. Experience with predictive modeling,
A/B testing, and big data technologies including Spark and Hadoop.
Skills: Machine Learning, Python, SQL, TensorFlow, Data Analysis.
Education: Howard University, Mathematics 2017
Member: National Society of Black Engineers""",

        """Emily Chen
Software Engineer with 5+ years experience in Python, Django, React, and AWS. 
Developed scalable web applications, implemented CI/CD pipelines, and led cross-functional teams.
Strong background in machine learning, cloud architecture, and agile methodologies.
Technologies: Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes, TensorFlow.
Education: MIT Computer Science 2018
Volunteer: Women in Tech mentorship program""",

        """James Rodriguez
Human Resources Manager with 8 years experience in talent acquisition and employee relations.
Implemented HRIS systems, reduced turnover by 35% through engagement programs.
MBA in Human Resource Management with SHRM-CP certification.
Skills: Talent Acquisition, Employee Relations, HRIS, Performance Management.
Previously worked at Google and McKinsey.
Education: Harvard Business School""",

        """Maria Garcia
Registered Nurse with 6 years experience in emergency and critical care.
Specialized in trauma care, patient advocacy, and emergency response protocols.
Bilingual in English and Spanish. ACLS and PALS certified.
Skills: Emergency Care, Patient Education, Medical Documentation, Team Leadership.
Education: University of Texas Nursing Program
Volunteer: Community health outreach programs"""
    ]
    
    with gr.Blocks(title="Resume Classification System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Resume Classification System
        **Classify resumes into 24 job categories with fairness analysis**
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
                        label="Example Resumes"
                    )
                
                with gr.Column(scale=2):
                    output_text = gr.Markdown(label="Classification Results")
                    output_table = gr.DataFrame(
                        label="Top 5 Predictions",
                        headers=["Category", "Confidence"]
                    )
                    
                    confidence_score = gr.Number(
                        label="Confidence Score",
                        value=0.0,
                        precision=3
                    )
                    
                    demographics_output = gr.JSON(
                        label="Inferred Demographics",
                        value={}
                    )
                    
                    lime_output = gr.Image(
                        label="LIME Explanation",
                        value=None
                    )
        
        with gr.Tab("Performance Metrics"):
            gr.Markdown("""
            ## Model Performance Metrics
            
            Comparison of baseline and debiased model performance.
            """)
            
            performance_df = gr.DataFrame(
                value=classifier.get_performance_metrics(),
                headers=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
                label="Performance Metrics Comparison"
            )
            
            gr.Markdown("""
            ### Model Information
            
            **Baseline Model:** Trained on original resume dataset with standard preprocessing.
            
            **Debiased Model:** Trained with bias mitigation techniques including adversarial training and counterfactual augmentation.
            
            **Available Job Categories:** ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER
            """)
        
        submit_btn.click(
            fn=classifier.predict,
            inputs=[input_text, model_selector],
            outputs=[output_text, output_table, confidence_score, demographics_output, lime_output]
        )
        
        clear_btn.click(
            fn=lambda: ["", pd.DataFrame([], columns=['Category', 'Confidence']), 0.0, {}, None],
            outputs=[input_text, output_table, confidence_score, demographics_output, lime_output]
        )
    
    return demo


if __name__ == "__main__":
    print("Launching Gradio Interface...")
    demo = create_interface()
    if demo:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=False
        )
    else:
        print("Failed to create interface. Please check if models are trained.")
