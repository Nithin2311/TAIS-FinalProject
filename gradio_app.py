"""
Gradio web interface for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Midterm Project
"""

import gradio as gr
import torch
import json
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processor import ResumePreprocessor


class ResumeClassifier:
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
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Please run train.py first to train the model")
            raise
    
    def predict(self, text):
        """Classify resume text and return predictions"""
        if not text or len(text.strip()) < 50:
            return "⚠️ Please enter at least 50 characters of resume text.", None, None
        
        try:
            # Preprocess
            cleaned_text = self.cleaner.clean_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
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
            result_text = "## 🎯 Classification Results\n\n"
            
            # Top prediction
            top_idx = top_indices[0].item()
            top_category = self.label_map[str(top_idx)]
            top_confidence = top_probs[0].item()
            
            result_text += f"**Primary Prediction:** {top_category}\n"
            result_text += f"**Confidence:** {top_confidence*100:.1f}%\n\n"
            
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
                category = self.label_map[str(idx.item())]
                confidence = prob.item() * 100
                result_text += f"{i}. **{category}**: {confidence:.1f}%\n"
                predictions_data.append([category, f"{confidence:.1f}%"])
            
            # Create DataFrame for table
            df = pd.DataFrame(predictions_data, columns=['Category', 'Confidence'])
            
            return result_text, df, top_confidence
            
        except Exception as e:
            return f"❌ Error during prediction: {str(e)}", None, None


def create_interface():
    """Create and launch Gradio interface"""
    
    # Check if model exists
    if not os.path.exists('models/resume_classifier'):
        print("❌ Model not found! Please run train.py first.")
        return None
    
    # Initialize classifier
    classifier = ResumeClassifier()
    
    # Example resumes
    examples = [
        """Software Engineer with 5+ years experience in Python, Django, React, and AWS. 
        Developed scalable web applications, implemented CI/CD pipelines, and led cross-functional teams.
        Strong background in machine learning, cloud architecture, and agile methodologies.""",
        
        """Human Resources Manager with 8 years experience in talent acquisition and employee relations.
        Implemented HRIS systems, reduced turnover by 35% through engagement programs.
        MBA in Human Resource Management with SHRM-CP certification.""",
        
        """Data Scientist specializing in machine learning and statistical analysis.
        Proficient in Python, R, SQL, TensorFlow, and Tableau. Experience with predictive modeling,
        A/B testing, and big data technologies including Spark and Hadoop.""",
        
        """Marketing Manager with 6+ years in digital marketing strategy.
        Expertise in SEO, SEM, social media marketing, and brand development.
        Increased online engagement by 200% and reduced CAC by 30%."""
    ]
    
    # Create Gradio interface
    with gr.Blocks(title="Resume Classifier - Midterm Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 AI-Powered Resume Classification System
        ## CAI 6605: Trustworthy AI Systems - Midterm Project
        **Group 15:** Nithin Palyam, Lorenzo LaPlace | **Date:** Fall 2025
        
        ### 📊 Model Performance
        - **Test Accuracy:** 84.45% (Target: >80%) ✅
        - **Model:** RoBERTa-base (125M parameters)
        - **Categories:** 24 job types
        - **Training Samples:** 2,484 resumes
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="📄 Resume Text Input",
                    placeholder="Paste resume content here (minimum 50 characters)...",
                    lines=10,
                    max_lines=20
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🎯 Classify Resume", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Examples(
                    examples=examples,
                    inputs=input_text,
                    label="💡 Example Resumes (Click to try)"
                )
            
            with gr.Column(scale=2):
                output_text = gr.Markdown(label="Results")
                output_table = gr.DataFrame(
                    label="Top 5 Predictions",
                    headers=["Category", "Confidence"]
                )
                confidence_score = gr.Number(
                    label="Confidence Score",
                    value=0.0,
                    precision=3
                )
        
        gr.Markdown("""
        ---
        ### 📋 Available Job Categories
        ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION,
        BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT,
        DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE,
        HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER
        
        ### 🎯 Key Features
        - Real-time classification into 24 job categories
        - Confidence scores for top predictions
        - Automated dataset download from Google Drive
        - Modular architecture ready for bias detection (final project)
        """)
        
        # Connect buttons
        submit_btn.click(
            fn=classifier.predict,
            inputs=input_text,
            outputs=[output_text, output_table, confidence_score]
        )
        
        clear_btn.click(
            fn=lambda: ["", None, 0.0],
            outputs=[input_text, output_table, confidence_score]
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Launching Gradio Interface...")
    demo = create_interface()
    if demo:
        demo.launch(share=True)
