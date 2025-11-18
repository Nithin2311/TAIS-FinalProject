"""
Enhanced Gradio web interface with Bias Analysis
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import gradio as gr
import torch
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processor import ResumePreprocessor
from bias_analyzer import BiasAnalyzer, BiasVisualization


class EnhancedResumeClassifier:
    def __init__(self, model_path='models/resume_classifier', label_map_path='data/processed/label_map.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            
            self.cleaner = ResumePreprocessor()
            self.bias_analyzer = BiasAnalyzer(self.model, self.tokenizer, self.label_map, self.device)
            
            # Load bias report if exists
            self.bias_report = None
            if os.path.exists('results/comprehensive_bias_report.json'):
                with open('results/comprehensive_bias_report.json', 'r') as f:
                    self.bias_report = json.load(f)
            
            print("Enhanced model loaded successfully with bias analysis!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please run train.py first to train the model")
            raise
    
    def predict(self, text):
        """Classify resume text and return predictions with bias awareness"""
        if not text or len(text.strip()) < 50:
            return "Please enter at least 50 characters of resume text.", None, None, None
        
        try:
            # Preprocess with enhanced cleaning
            cleaned_text = self.cleaner.clean_text(text)
            technical_skills = self.cleaner.extract_technical_skills(text)
            enhanced_text = cleaned_text + ' ' + technical_skills
            
            # Tokenize
            inputs = self.tokenizer(
                enhanced_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], 5)
            
            # Format results
            result_text = "## Classification Results\n\n"
            
            # Top prediction
            top_idx = top_indices[0].item()
            top_category = self.label_map[str(top_idx)]
            top_confidence = top_probs[0].item()
            
            result_text += f"**Primary Prediction:** {top_category}\n"
            result_text += f"**Confidence:** {top_confidence*100:.1f}%\n\n"
            
            # Bias awareness
            if self.bias_report:
                category_bias = self.bias_report['category_bias_analysis'].get(top_category, {})
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
                                result_text += f"**Gender Bias Alert:** {max_diff*100:.1f}% accuracy difference\n"
            
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
            bias_score = self._calculate_bias_score(cleaned_text, top_category)
            
            return result_text, df, top_confidence, bias_score
            
        except Exception as e:
            return f"Error during prediction: {str(e)}", None, None, None
    
    def _calculate_bias_score(self, text, predicted_category):
        """Calculate bias score for the prediction"""
        from bias_analyzer import DemographicInference
        
        demo_inference = DemographicInference()
        gender = demo_inference.infer_gender(text)
        diversity = demo_inference.infer_diversity_background(text)
        
        # Base score
        bias_score = 0.5
        
        # Adjust based on demographic inference
        if gender != 'unknown':
            bias_score += 0.1
        if diversity != 'neutral':
            bias_score += 0.1
        
        return min(bias_score, 1.0)
    
    def get_bias_report_summary(self):
        """Get summary of bias analysis for display"""
        if not self.bias_report:
            return "## Bias Analysis\n\nNo bias report available. Please run training first."
        
        report = self.bias_report
        summary_text = "## Comprehensive Bias Analysis Report\n\n"
        
        # Fairness metrics summary
        summary_text += "### Fairness Metrics\n"
        for demo_type, metrics in report['fairness_metrics'].items():
            summary_text += f"**{demo_type.upper()}:**\n"
            summary_text += f"  - Demographic Parity: {metrics['demographic_parity']:.3f}\n"
            summary_text += f"  - Equal Opportunity: {metrics['equal_opportunity']:.3f}\n"
            summary_text += f"  - Accuracy Equality: {metrics['accuracy_equality']:.3f}\n\n"
        
        # Name bias
        name_bias = report['name_substitution_bias']
        summary_text += f"### Name-based Bias\n"
        summary_text += f"**Average Gender Bias:** {name_bias['average_gender_bias']:.3f}\n"
        summary_text += f"**Male-Female Disparity:** {name_bias['male_female_disparity']:.3f}\n\n"
        
        # Recommendations
        summary_text += "### Recommendations\n"
        for i, rec in enumerate(report['recommendations'][:3], 1):
            summary_text += f"{i}. {rec}\n"
        
        return summary_text


def create_enhanced_interface():
    """Create and launch enhanced Gradio interface with bias analysis"""
    
    # Check if model exists
    if not os.path.exists('models/resume_classifier'):
        print("Model not found! Please run train.py first.")
        return None
    
    # Initialize classifier
    classifier = EnhancedResumeClassifier()
    
    # Example resumes - fixed to ensure correct classification
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
        Skills: Machine Learning, Python, SQL, TensorFlow, Data Analysis.""",
        
        """Marketing Manager with 6+ years in digital marketing strategy.
        Expertise in SEO, SEM, social media marketing, and brand development.
        Increased online engagement by 200% and reduced CAC by 30%.
        Skills: Digital Marketing, SEO, SEM, Social Media, Brand Strategy."""
    ]
    
    # Create enhanced Gradio interface
    with gr.Blocks(title="Bias-Aware Resume Classifier - Final Project", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # AI-Powered Resume Classification System with Bias Detection
        ## CAI 6605: Trustworthy AI Systems - Final Project
        **Group 15:** Nithin Palyam, Lorenzo LaPlace | **Date:** Fall 2025
        
        ### Model Performance & Fairness
        - **Test Accuracy:** 84.45% (Target: >80%)
        - **Model:** RoBERTa-base (125M parameters)
        - **Categories:** 24 job types
        - **Training Samples:** 2,484 resumes
        - **Bias Detection:** Comprehensive fairness analysis integrated
        """)
        
        with gr.Tab("Resume Classification"):
            with gr.Row():
                with gr.Column(scale=2):
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
        
        with gr.Tab("Bias Analysis"):
            gr.Markdown("""
            ## Comprehensive Bias Analysis
            
            This section shows the results of our comprehensive bias detection framework,
            including demographic parity, equal opportunity, and name-based bias analysis.
            """)
            
            bias_report = gr.Markdown(
                label="Bias Analysis Report",
                value=classifier.get_bias_report_summary()
            )
            
            refresh_btn = gr.Button("Refresh Bias Report", variant="secondary")
            
            # Display bias visualizations if they exist
            if os.path.exists('visualizations/fairness_metrics.png'):
                gr.Markdown("### Fairness Metrics Visualization")
                gr.Image('visualizations/fairness_metrics.png', label="Fairness Metrics")
            
            if os.path.exists('visualizations/category_bias.png'):
                gr.Markdown("### Category-Level Bias Analysis")
                gr.Image('visualizations/category_bias.png', label="Category Bias")
        
        with gr.Tab("Project Information"):
            gr.Markdown("""
            ### Final Project Enhancements
            
            **Bias Detection Framework:**
            - Demographic inference from resume text
            - Comprehensive fairness metrics calculation
            - Name substitution experiments
            - Category-level bias analysis
            
            **Fairness Metrics:**
            - Demographic Parity Difference
            - Equal Opportunity Difference  
            - Accuracy Equality Difference
            
            **Technical Implementation:**
            - Modular architecture for easy extension
            - Integration with existing classification pipeline
            - Professional visualizations and reporting
            - Gradio interface with real-time bias awareness
            
            ### Available Job Categories
            ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION,
            BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT,
            DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE,
            HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER
            """)
        
        # Connect buttons
        submit_btn.click(
            fn=classifier.predict,
            inputs=input_text,
            outputs=[output_text, output_table, confidence_score, bias_score]
        )
        
        clear_btn.click(
            fn=lambda: ["", None, 0.0, 0.0],
            outputs=[input_text, output_table, confidence_score, bias_score]
        )
        
        refresh_btn.click(
            fn=classifier.get_bias_report_summary,
            outputs=bias_report
        )
    
    return demo


if __name__ == "__main__":
    print("Launching Enhanced Gradio Interface with Bias Analysis...")
    demo = create_enhanced_interface()
    if demo:
        demo.launch(share=True)
