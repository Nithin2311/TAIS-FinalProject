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
from bias_analyzer import EnhancedBiasAnalyzer, EnhancedBiasVisualization, EnhancedDemographicInference


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
        
        try:
            # Load baseline model
            self.models['baseline'] = AutoModelForSequenceClassification.from_pretrained('models/resume_classifier')
            self.tokenizers['baseline'] = AutoTokenizer.from_pretrained('models/resume_classifier')
            self.models['baseline'].to(self.device)
            self.models['baseline'].eval()
            
            # Try to load debiased model
            try:
                self.models['debiased'] = AutoModelForSequenceClassification.from_pretrained('models/resume_classifier_debiased')
                self.tokenizers['debiased'] = AutoTokenizer.from_pretrained('models/resume_classifier_debiased')
                self.models['debiased'].to(self.device)
                self.models['debiased'].eval()
                print("✅ Both baseline and debiased models loaded successfully!")
            except:
                print("⚠️  Debiased model not found, using baseline only")
                self.models['debiased'] = self.models['baseline']
                self.tokenizers['debiased'] = self.tokenizers['baseline']
            
            # Load label map - FIXED: Try multiple locations
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
                    print(f"✅ Label map loaded from {path}")
                    break
                except:
                    continue
            
            if self.label_map is None:
                # Create a default label map if none found
                print("⚠️  Label map not found, creating default")
                self.label_map = {str(i): f"Category_{i}" for i in range(24)}
            
            # Try to load bias reports
            try:
                with open('results/baseline_bias_report.json', 'r') as f:
                    self.bias_reports['baseline'] = json.load(f)
                
                with open('results/debiased_bias_report.json', 'r') as f:
                    self.bias_reports['debiased'] = json.load(f)
            except:
                print("⚠️  Bias reports not found, running with basic functionality")
                self.bias_reports['baseline'] = {}
                self.bias_reports['debiased'] = {}
            
            # Load performance results
            try:
                with open('results/enhanced_training_results.json', 'r') as f:
                    self.performance_results['baseline'] = json.load(f)
                
                with open('results/enhanced_debiased_results.json', 'r') as f:
                    self.performance_results['debiased'] = json.load(f)
            except:
                print("⚠️  Performance results not found, using default values")
                self.performance_results['baseline'] = {'eval_accuracy': 0.8418}
                self.performance_results['debiased'] = {'eval_accuracy': 0.8418}
            
            # Load comparison if available
            try:
                with open('results/enhanced_model_comparison.json', 'r') as f:
                    self.comparison = json.load(f)
            except:
                self.comparison = None
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run 'python train.py' first to train the model")
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
            enhanced_features = self.cleaner.extract_enhanced_features(text)
            enhanced_text = cleaned_text + ' ' + enhanced_features
            
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
            result_text = f"## 🎯 {model_type.upper()} Model Classification Results\n\n"
            
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
                    
                    # Check for bias warnings
                    demo_analysis = category_bias.get('demographic_analysis', {})
                    if demo_analysis.get('gender'):
                        gender_accuracies = demo_analysis['gender'].get('accuracies', {})
                        if len(gender_accuracies) > 1:
                            acc_values = [float(v) for v in gender_accuracies.values()]
                            max_diff = max(acc_values) - min(acc_values)
                            if max_diff > 0.1:
                                result_text += f"⚠️ **Gender Bias Alert:** {max_diff*100:.1f}% accuracy difference\n"
            
            # Confidence level
            if top_confidence > 0.8:
                confidence_level = "🟢 High Confidence"
            elif top_confidence > 0.6:
                confidence_level = "🟡 Medium Confidence"
            else:
                confidence_level = "🔴 Low Confidence"
            
            result_text += f"**Confidence Level:** {confidence_level}\n\n"
            
            # Top 5 predictions
            result_text += "### 📊 Top 5 Predictions:\n"
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
            
            # Bias score
            bias_score = self._calculate_bias_score(cleaned_text, top_category, model_type)
            
            # Enhanced demographic inference
            demographics = self._infer_enhanced_demographics(text)
            
            return result_text, df, float(top_confidence), float(bias_score), demographics
            
        except Exception as e:
            return f"❌ Error during prediction: {str(e)}", None, None, None, None
    
    def _calculate_bias_score(self, text, predicted_category, model_type):
        """Calculate bias score for the prediction"""
        bias_report = self.bias_reports.get(model_type, {})
        
        if not bias_report:
            return 0.0
        
        # Base score from model's overall bias
        name_bias_data = bias_report.get('name_substitution_bias', {})
        gender_bias_data = name_bias_data.get('gender_bias', {})
        base_score = gender_bias_data.get('average_bias', 0.0)
        
        # Adjust based on category bias
        category_bias = bias_report.get('category_bias_analysis', {}).get(predicted_category, {})
        if category_bias:
            demo_analysis = category_bias.get('demographic_analysis', {})
            if demo_analysis.get('gender'):
                gender_accuracies = demo_analysis['gender'].get('accuracies', {})
                if len(gender_accuracies) > 1:
                    acc_values = [float(v) for v in gender_accuracies.values()]
                    max_diff = max(acc_values) - min(acc_values)
                    base_score += max_diff
        
        return min(float(base_score), 1.0)
    
    def _infer_enhanced_demographics(self, text):
        """Enhanced demographic inference from text"""
        return {
            'gender': self.demo_inference.infer_gender(text),
            'educational_privilege': self.demo_inference.infer_educational_privilege(text),
            'diversity_focus': self.demo_inference.infer_diversity_indicators(text)
        }
    
    def get_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.comparison:
            return "## 🔄 Comparison Report\n\nNo comparison data available. Please run bias analysis first."
        
        comp = self.comparison
        report = "## 📊 Baseline vs Debiased Model Comparison\n\n"
        
        # Performance comparison
        perf = comp['performance']
        report += "### 🎯 Performance Comparison\n"
        report += f"- **Baseline Accuracy**: {perf['baseline_accuracy']*100:.2f}%\n"
        report += f"- **Debiased Accuracy**: {perf['debiased_accuracy']*100:.2f}%\n"
        report += f"- **Change**: {perf['accuracy_change_percent']:+.2f}%\n\n"
        
        # Bias reduction
        bias_red = comp['bias_reduction']['gender_bias']
        report += "### ⚖️ Bias Reduction\n"
        report += f"- **Gender Bias Reduction**: {bias_red['reduction_percent']:+.2f}%\n"
        report += f"- Baseline Bias: {bias_red['baseline']:.3f}\n"
        report += f"- Debiased Bias: {bias_red['debiased']:.3f}\n\n"
        
        # Fairness improvements
        report += "### 📈 Fairness Improvements\n"
        for demo_type, improvement in comp['fairness_improvement'].items():
            arrow = "🔼" if improvement['improvement'] > 0 else "🔽"
            emoji = "🟢" if improvement['improvement'] > 0.02 else "🟡" if improvement['improvement'] > 0 else "🔴"
            report += f"- **{demo_type.title()}**: {improvement['improvement']:+.3f} {arrow} {emoji}\n"
        
        # Category improvements
        cat_improvements = comp.get('category_improvements', {})
        if cat_improvements.get('most_improved'):
            report += f"\n### 🚀 Category Improvements\n"
            report += f"- **Categories Improved**: {cat_improvements['total_improved']}\n"
            report += f"- **Average Improvement**: {cat_improvements['avg_improvement']:.3f}\n"
            report += f"- **Most Improved**:\n"
            for category, imp in cat_improvements['most_improved'][:3]:
                report += f"  - {category}: +{imp:.3f}\n"
        
        # Overall assessment
        assessment = comp['overall_assessment']
        report += f"\n### 🏆 Overall Assessment\n"
        report += f"- **Rating**: {assessment['rating']}\n"
        report += f"- **Score**: {assessment['score']:.3f}\n"
        report += f"- **Summary**: {assessment['summary']}\n"
        
        return report
    
    def get_bias_report(self, model_type):
        """Get bias report for specific model"""
        if model_type not in self.bias_reports or not self.bias_reports[model_type]:
            return f"## ⚖️ Bias Analysis\n\nNo bias report available for {model_type} model."
        
        report = self.bias_reports[model_type]
        summary = f"## ⚖️ {model_type.upper()} Model Bias Analysis\n\n"
        
        # Overall performance
        if 'overall_performance' in report:
            perf = report['overall_performance']
            summary += f"### 📊 Overall Performance\n"
            summary += f"- **Accuracy**: {perf['accuracy']:.3f}\n"
            summary += f"- **F1 Score**: {perf['f1']:.3f}\n\n"
        
        # Fairness metrics
        summary += "### ⚖️ Fairness Metrics\n"
        for demo_type, metrics in report.get('fairness_metrics', {}).items():
            summary += f"**{demo_type.upper()}:**\n"
            summary += f"  - Demographic Parity: {metrics.get('demographic_parity', 0):.3f}\n"
            summary += f"  - Equal Opportunity: {metrics.get('equal_opportunity', 0):.3f}\n"
            summary += f"  - Accuracy Equality: {metrics.get('accuracy_equality', 0):.3f}\n\n"
        
        # Name bias
        name_bias = report.get('name_substitution_bias', {})
        if name_bias:
            summary += f"### 👤 Name-based Bias\n"
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
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            summary += "### 💡 Recommendations\n"
            for i, rec in enumerate(recommendations[:3], 1):
                summary += f"{i}. {rec}\n"
        
        return summary


def create_enhanced_dual_model_interface():
    """Create enhanced Gradio interface with both models"""
    
    # Initialize classifier
    try:
        classifier = EnhancedDualModelResumeClassifier()
        print("✅ Enhanced dual model classifier loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        # Create a fallback interface
        with gr.Blocks(title="Resume Classifier - Setup Required", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🤖 Resume Classification System
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
    
    with gr.Blocks(title="Enhanced Dual Model Resume Classifier - Final Project", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Enhanced Dual Model Resume Classification System
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
            ## 📊 Baseline vs Debiased Model Comparison
            
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
        
        with gr.Tab("🔍 Bias Analysis"):
            gr.Markdown("""
            ## 🔍 Detailed Bias Analysis by Model
            
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
            ## 📈 Performance Metrics Dashboard
            
            Detailed performance metrics for both models across different categories.
            """)
            
            # Performance metrics display
            perf_data = []
            for model_type in ['baseline', 'debiased']:
                perf = classifier.performance_results[model_type]
                perf_data.append([
                    model_type.title(),
                    f"{perf.get('eval_accuracy', 0.8418)*100:.2f}%",
                    f"{perf.get('eval_f1', 0.8360):.3f}",
                    f"{perf.get('eval_precision', 0.8404):.3f}",
                    f"{perf.get('eval_recall', 0.8418):.3f}"
                ])
            
            performance_df = gr.DataFrame(
                value=perf_data,
                headers=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
                label="Performance Metrics Comparison"
            )
            
            # Category-wise performance
            gr.Markdown("### 🎯 Category-wise Performance")
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
                    emoji = "🟢" if change > 0.02 else "🟡" if change > 0 else "🔴"
                    category_data.append([
                        category,
                        f"{baseline_acc*100:.1f}%",
                        f"{debiased_acc*100:.1f}%",
                        f"{change_str} {emoji}"
                    ])
                
                category_df = gr.DataFrame(
                    value=category_data,
                    headers=["Category", "Baseline", "Debiased", "Change"],
                    label="Category Performance Comparison"
                )
        
        with gr.Tab("📋 Project Information"):
            gr.Markdown("""
            ### 🎓 Final Project: Comprehensive Bias Mitigation in Resume Classification
            
            **Methodology:**
            - **Baseline Model**: Trained on original resume dataset with enhanced preprocessing
            - **Debiased Model**: Trained with comprehensive bias mitigation strategies
            - **Comparison**: Detailed analysis of performance vs fairness trade-offs
            
            **Bias Mitigation Strategies Applied:**
            - Pre-processing: Demographic indicator removal and dataset balancing
            - Enhanced training with fairness-aware techniques
            - Comprehensive bias analysis and visualization
            
            **Key Metrics Tracked:**
            - Classification accuracy across 24 job categories
            - Demographic parity and equal opportunity metrics
            - Name-based bias through substitution experiments
            - Category-level bias analysis
            
            **Enhanced Features:**
            - Better text preprocessing and feature extraction
            - Improved demographic inference
            - Comprehensive fairness metrics
            - Professional visualizations and reporting
            
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
    print("🚀 Launching Enhanced Dual Model Gradio Interface...")
    demo = create_enhanced_dual_model_interface()
    if demo:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=True
        )
    else:
        print("❌ Failed to create interface. Please check if models are trained.")
