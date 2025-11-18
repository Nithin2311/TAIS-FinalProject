"""
Enhanced Gradio web interface with Dual Model Comparison
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
from bias_analyzer import BiasAnalyzer, BiasVisualization, DemographicInference


class DualModelResumeClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cleaner = ResumePreprocessor()
        self.demo_inference = DemographicInference()
        
        # Load both models
        self.models = {}
        self.tokenizers = {}
        self.bias_reports = {}
        self.performance_results = {}
        
        try:
            # Load baseline model
            self.models['baseline'] = AutoModelForSequenceClassification.from_pretrained('models/resume_classifier')
            self.tokenizers['baseline'] = AutoTokenizer.from_pretrained('models/resume_classifier')
            self.models['baseline'].to(self.device)
            self.models['baseline'].eval()
            
            # Load debiased model
            self.models['debiased'] = AutoModelForSequenceClassification.from_pretrained('models/resume_classifier_debiased')
            self.tokenizers['debiased'] = AutoTokenizer.from_pretrained('models/resume_classifier_debiased')
            self.models['debiased'].to(self.device)
            self.models['debiased'].eval()
            
            # Load label map
            with open('data/processed/label_map.json', 'r') as f:
                self.label_map = json.load(f)
            
            # Load bias reports
            with open('results/baseline_bias_report.json', 'r') as f:
                self.bias_reports['baseline'] = json.load(f)
            
            with open('results/debiased_bias_report.json', 'r') as f:
                self.bias_reports['debiased'] = json.load(f)
            
            # Load performance results
            with open('results/training_results.json', 'r') as f:
                self.performance_results['baseline'] = json.load(f)
            
            with open('results/debiased_results.json', 'r') as f:
                self.performance_results['debiased'] = json.load(f)
            
            # Load comparison
            with open('results/model_comparison.json', 'r') as f:
                self.comparison = json.load(f)
            
            print("✅ Dual model classifier loaded successfully!")
            print("✅ Both baseline and debiased models are available!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run 'python bias_analysis.py' first to train both models and generate analysis")
            raise
    
    def predict(self, text, model_type='baseline'):
        """Classify resume text with specified model"""
        if not text or len(text.strip()) < 50:
            return "Please enter at least 50 characters of resume text.", None, None, None, None
        
        try:
            model = self.models[model_type]
            tokenizer = self.tokenizers[model_type]
            
            # Preprocess text
            cleaned_text = self.cleaner.clean_text(text)
            technical_skills = self.cleaner.extract_technical_skills(text)
            enhanced_text = cleaned_text + ' ' + technical_skills
            
            # Tokenize and predict
            inputs = tokenizer(
                enhanced_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], 5)
            
            # Format results
            result_text = f"## {model_type.upper()} Model Classification Results\n\n"
            
            # Top prediction
            top_idx = top_indices[0].item()
            top_category = self.label_map[str(top_idx)]
            top_confidence = top_probs[0].item()
            
            result_text += f"**Primary Prediction:** {top_category}\n"
            result_text += f"**Confidence:** {top_confidence*100:.1f}%\n\n"
            
            # Model performance info
            model_perf = self.performance_results[model_type]
            result_text += f"**Model Accuracy:** {model_perf['eval_accuracy']*100:.1f}%\n"
            
            # Bias awareness
            bias_report = self.bias_reports[model_type]
            category_bias = bias_report['category_bias_analysis'].get(top_category, {})
            if category_bias:
                overall_accuracy = category_bias.get('overall_accuracy', 0)
                result_text += f"**Category Accuracy:** {overall_accuracy*100:.1f}%\n"
                
                # Check for bias warnings
                demo_analysis = category_bias.get('demographic_analysis', {})
                if demo_analysis.get('gender'):
                    gender_accuracies = demo_analysis['gender']
                    if len(gender_accuracies) > 1:
                        acc_values = list(gender_accuracies.values())
                        max_diff = max(acc_values) - min(acc_values)
                        if max_diff > 0.1:
                            result_text += f"⚠️ **Gender Bias Alert:** {max_diff*100:.1f}% accuracy difference\n"
            
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
                category = self.label_map[str(idx.item())]
                confidence = prob.item() * 100
                result_text += f"{i}. **{category}**: {confidence:.1f}%\n"
                predictions_data.append([category, f"{confidence:.1f}%"])
            
            # Create DataFrame for table
            df = pd.DataFrame(predictions_data, columns=['Category', 'Confidence'])
            
            # Bias score
            bias_score = self._calculate_bias_score(cleaned_text, top_category, model_type)
            
            # Demographic inference
            demographics = self._infer_demographics(text)
            
            return result_text, df, top_confidence, bias_score, demographics
            
        except Exception as e:
            return f"Error during prediction: {str(e)}", None, None, None, None
    
    def _calculate_bias_score(self, text, predicted_category, model_type):
        """Calculate bias score for the prediction"""
        bias_report = self.bias_reports[model_type]
        
        # Base score from model's overall bias
        name_bias = bias_report['name_substitution_bias']['average_gender_bias']
        base_score = name_bias
        
        # Adjust based on category bias
        category_bias = bias_report['category_bias_analysis'].get(predicted_category, {})
        if category_bias:
            demo_analysis = category_bias.get('demographic_analysis', {})
            if demo_analysis.get('gender'):
                gender_accuracies = demo_analysis['gender']
                if len(gender_accuracies) > 1:
                    max_diff = max(gender_accuracies.values()) - min(gender_accuracies.values())
                    base_score += max_diff
        
        return min(base_score, 1.0)
    
    def _infer_demographics(self, text):
        """Infer demographics from text"""
        return {
            'gender': self.demo_inference.infer_gender(text),
            'diversity_background': self.demo_inference.infer_diversity_background(text),
            'privilege_level': self.demo_inference.infer_privilege_level(text)
        }
    
    def get_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not hasattr(self, 'comparison'):
            return "## Comparison Report\n\nNo comparison data available."
        
        comp = self.comparison
        report = "## Baseline vs Debiased Model Comparison\n\n"
        
        # Performance comparison
        perf = comp['performance']
        report += "### 📊 Performance Comparison\n"
        report += f"- **Baseline Accuracy**: {perf['baseline_accuracy']*100:.2f}%\n"
        report += f"- **Debiased Accuracy**: {perf['debiased_accuracy']*100:.2f}%\n"
        report += f"- **Change**: {perf['accuracy_change_percent']:+.2f}%\n\n"
        
        # Bias reduction
        bias_red = comp['bias_reduction']['name_bias']
        report += "### ⚖️ Bias Reduction\n"
        report += f"- **Name-based Bias Reduction**: {bias_red['reduction_percent']:+.2f}%\n"
        report += f"- Baseline Bias: {bias_red['baseline']:.3f}\n"
        report += f"- Debiased Bias: {bias_red['debiased']:.3f}\n\n"
        
        # Fairness improvements
        report += "### 📈 Fairness Improvements\n"
        for demo_type, improvement in comp['fairness_improvement'].items():
            arrow = "↗️" if improvement['improvement'] > 0 else "↘️"
            report += f"- **{demo_type.title()}**: {improvement['improvement']:+.3f} {arrow}\n"
        
        # Overall assessment
        assessment = comp['overall_assessment']
        report += f"\n### 🎯 Overall Assessment\n"
        report += f"- **Rating**: {assessment['rating']}\n"
        report += f"- **Score**: {assessment['score']:.3f}\n"
        report += f"- **Summary**: {assessment['summary']}\n"
        
        return report
    
    def get_bias_report(self, model_type):
        """Get bias report for specific model"""
        if model_type not in self.bias_reports:
            return f"## Bias Analysis\n\nNo bias report available for {model_type} model."
        
        report = self.bias_reports[model_type]
        summary = f"## {model_type.upper()} Model Bias Analysis\n\n"
        
        # Fairness metrics
        summary += "### Fairness Metrics\n"
        for demo_type, metrics in report['fairness_metrics'].items():
            summary += f"**{demo_type.upper()}:**\n"
            summary += f"  - Demographic Parity: {metrics['demographic_parity']:.3f}\n"
            summary += f"  - Equal Opportunity: {metrics['equal_opportunity']:.3f}\n"
            summary += f"  - Accuracy Equality: {metrics['accuracy_equality']:.3f}\n\n"
        
        # Name bias
        name_bias = report['name_substitution_bias']
        summary += f"### Name-based Bias\n"
        summary += f"**Average Gender Bias:** {name_bias['average_gender_bias']:.3f}\n"
        summary += f"**Male-Female Disparity:** {name_bias['male_female_disparity']:.3f}\n\n"
        
        # Recommendations
        summary += "### Recommendations\n"
        for i, rec in enumerate(report['recommendations'][:3], 1):
            summary += f"{i}. {rec}\n"
        
        return summary


def create_dual_model_interface():
    """Create Gradio interface with both models"""
    
    # Initialize classifier
    classifier = DualModelResumeClassifier()
    
    # Example resumes
    examples = [
        """Software Engineer with 5+ years experience in Python, Django, React, and AWS. 
        Developed scalable web applications, implemented CI/CD pipelines, and led cross-functional teams.
        Strong background in machine learning, cloud architecture, and agile methodologies.
        Technologies: Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes, TensorFlow.""",
        
        """Human Resources Manager with 8 years experience in talent acquisition and employee relations.
        Implemented HRIS systems, reduced turnover by 35% through engagement programs.
        MBA in Human Resource Management with SHRM-CP certification.
        Skills: Talent Acquisition, Employee Relations, HRIS, Performance Management.""",
        
        """Data Scientist specializing in machine learning and statistical analysis.
        Proficient in Python, R, SQL, TensorFlow, and Tableau. Experience with predictive modeling,
        A/B testing, and big data technologies including Spark and Hadoop.
        Skills: Machine Learning, Python, SQL, TensorFlow, Data Analysis."""
    ]
    
    with gr.Blocks(title="Dual Model Resume Classifier - Final Project", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Dual Model Resume Classification System
        ## CAI 6605: Trustworthy AI Systems - Final Project
        **Group 15:** Nithin Palyam, Lorenzo LaPlace  
        **Compare Baseline vs Debiased Models with Comprehensive Bias Analysis**
        """)
        
        with gr.Tab("🚀 Resume Classification"):
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
                        label="Example Resumes (Click to try)"
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
                        bias_score = gr.Number(
                            label="Bias Risk Score",
                            value=0.0,
                            precision=3
                        )
                    
                    demographics_output = gr.JSON(
                        label="Inferred Demographics",
                        value={}
                    )
        
        with gr.Tab("📊 Model Comparison"):
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
                if os.path.exists('visualizations/fairness_comparison.png'):
                    gr.Image('visualizations/fairness_comparison.png', 
                           label="Fairness Metrics Comparison")
                
                if os.path.exists('visualizations/performance_bias_comparison.png'):
                    gr.Image('visualizations/performance_bias_comparison.png', 
                           label="Performance & Bias Comparison")
        
        with gr.Tab("🔍 Bias Analysis"):
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
        
        with gr.Tab("📈 Performance Metrics"):
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
                    f"{perf['eval_accuracy']*100:.2f}%",
                    f"{perf['eval_f1']:.3f}",
                    f"{perf['eval_precision']:.3f}",
                    f"{perf['eval_recall']:.3f}"
                ])
            
            performance_df = gr.DataFrame(
                value=perf_data,
                headers=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
                label="Performance Metrics Comparison"
            )
            
            # Category-wise performance
            gr.Markdown("### Category-wise Performance (Top 10)")
            category_data = []
            baseline_perf = classifier.performance_results['baseline']['per_class_accuracy']
            debiased_perf = classifier.performance_results['debiased']['per_class_accuracy']
            
            # Get common categories and sort by baseline performance
            common_categories = set(baseline_perf.keys()) & set(debiased_perf.keys())
            sorted_categories = sorted(common_categories, 
                                     key=lambda x: baseline_perf[x], 
                                     reverse=True)[:10]
            
            for category in sorted_categories:
                baseline_acc = baseline_perf.get(category, 0)
                debiased_acc = debiased_perf.get(category, 0)
                category_data.append([
                    category,
                    f"{baseline_acc*100:.1f}%",
                    f"{debiased_acc*100:.1f}%",
                    f"{(debiased_acc - baseline_acc)*100:+.1f}%"
                ])
            
            category_df = gr.DataFrame(
                value=category_data,
                headers=["Category", "Baseline", "Debiased", "Change"],
                label="Top 10 Categories Performance Comparison"
            )
        
        with gr.Tab("📋 Project Information"):
            gr.Markdown("""
            ### Final Project: Comprehensive Bias Mitigation in Resume Classification
            
            **Methodology:**
            - **Baseline Model**: Trained on original resume dataset
            - **Debiased Model**: Trained with comprehensive bias mitigation strategies
            - **Comparison**: Detailed analysis of performance vs fairness trade-offs
            
            **Bias Mitigation Strategies Applied:**
            - Pre-processing: Demographic indicator removal and dataset balancing
            - Enhanced training with fairness-aware techniques
            
            **Key Metrics Tracked:**
            - Classification accuracy across 24 job categories
            - Demographic parity and equal opportunity metrics
            - Name-based bias through substitution experiments
            - Category-level bias analysis
            
            **Available Job Categories:**
            ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION,
            BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT,
            DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE,
            HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER
            """)
        
        # Connect buttons
        submit_btn.click(
            fn=classifier.predict,
            inputs=[input_text, model_selector],
            outputs=[output_text, output_table, confidence_score, bias_score, demographics_output]
        )
        
        clear_btn.click(
            fn=lambda: ["", None, 0.0, 0.0, {}],
            outputs=[input_text, output_table, confidence_score, bias_score, demographics_output]
        )
    
    return demo


if __name__ == "__main__":
    print("Launching Dual Model Gradio Interface...")
    demo = create_dual_model_interface()
    if demo:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860
        )
