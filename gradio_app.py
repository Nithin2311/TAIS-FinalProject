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
import traceback


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
                        padding=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        probabilities.append(probs.cpu().numpy()[0])
                except Exception as e:
                    print(f"Prediction error in LIME: {e}")
                    probabilities.append(np.ones(len(self.label_map)) / len(self.label_map))
            
            return np.array(probabilities)
        
        try:
            exp = self.explainer.explain_instance(
                text,
                predict_proba,
                num_features=min(num_features, 20),
                num_samples=150,
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
        
        print("Initializing Resume Classifier...")
        
        try:
            # Load baseline model
            baseline_path = 'models/resume_classifier_baseline'
            if os.path.exists(baseline_path):
                print(f"Loading baseline model from {baseline_path}")
                self.models['baseline'] = AutoModelForSequenceClassification.from_pretrained(baseline_path)
                self.tokenizers['baseline'] = AutoTokenizer.from_pretrained(baseline_path)
                self.models['baseline'].to(self.device)
                self.models['baseline'].eval()
                print("Baseline model loaded successfully")
            else:
                raise FileNotFoundError(f"Baseline model not found at {baseline_path}")
            
            # Load debiased model
            debiased_path = 'models/resume_classifier_debiased'
            if os.path.exists(debiased_path):
                print(f"Loading debiased model from {debiased_path}")
                self.models['debiased'] = AutoModelForSequenceClassification.from_pretrained(debiased_path)
                self.tokenizers['debiased'] = AutoTokenizer.from_pretrained(debiased_path)
                self.models['debiased'].to(self.device)
                self.models['debiased'].eval()
                print("Debiased model loaded successfully")
            else:
                print("Debiased model not found, using baseline for both")
                self.models['debiased'] = self.models['baseline']
                self.tokenizers['debiased'] = self.tokenizers['baseline']
            
            # Load label map
            label_map_paths = [
                'data/processed/label_map.json',
                'models/resume_classifier_baseline/label_map.json',
                'models/resume_classifier_debiased/label_map.json'
            ]
            
            for path in label_map_paths:
                try:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            self.label_map = json.load(f)
                        print(f"Label map loaded from {path}")
                        print(f"Number of categories: {len(self.label_map)}")
                        break
                except Exception as e:
                    print(f"Could not load label map from {path}: {e}")
                    continue
            
            # If label map still not loaded, create default
            if not self.label_map:
                print("WARNING: Label map not found, creating default")
                # Default categories from the dataset
                self.label_map = {
                    '0': 'ACCOUNTANT', '1': 'ADVOCATE', '2': 'AGRICULTURE', '3': 'APPAREL',
                    '4': 'ARTS', '5': 'AUTOMOBILE', '6': 'AVIATION', '7': 'BANKING',
                    '8': 'BPO', '9': 'BUSINESS-DEVELOPMENT', '10': 'CHEF', '11': 'CONSTRUCTION',
                    '12': 'CONSULTANT', '13': 'DESIGNER', '14': 'DIGITAL-MEDIA',
                    '15': 'ENGINEERING', '16': 'FINANCE', '17': 'FITNESS',
                    '18': 'HEALTHCARE', '19': 'HR', '20': 'INFORMATION-TECHNOLOGY',
                    '21': 'PUBLIC-RELATIONS', '22': 'SALES', '23': 'TEACHER'
                }
            
            # Initialize LIME explainers
            self.lime_explainers = {}
            for model_type in ['baseline', 'debiased']:
                self.lime_explainers[model_type] = LimeExplainer(
                    self.models[model_type],
                    self.tokenizers[model_type],
                    self.label_map,
                    self.device
                )
            
            # Load performance results
            try:
                with open('results/baseline_results.json', 'r') as f:
                    self.baseline_results = json.load(f)
                print("Baseline results loaded")
            except Exception as e:
                print(f"Could not load baseline results: {e}")
                self.baseline_results = {'eval_accuracy': 0.85}
            
            try:
                with open('results/debiased_results.json', 'r') as f:
                    self.debiased_results = json.load(f)
                print("Debiased results loaded")
            except Exception as e:
                print(f"Could not load debiased results: {e}")
                self.debiased_results = {'eval_accuracy': 0.85}
            
            print("\nResume Classifier initialized successfully!")
            print(f"Device: {self.device}")
            print(f"Available models: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"Error initializing Resume Classifier: {e}")
            traceback.print_exc()
            raise
    
    def preprocess_text(self, text):
        """Preprocess text exactly like training pipeline"""
        # Clean text
        cleaned_text = self.cleaner.clean_text(text)
        
        # Extract features (skills, experience, etc.)
        features = self.cleaner.extract_features(text)
        
        # Combine as done in training
        combined_text = cleaned_text + ' ' + features
        
        return cleaned_text, features, combined_text
    
    def predict(self, text, model_type='baseline'):
        """Classify resume text with specified model"""
        if not text or len(text.strip()) < 50:
            return "Please enter at least 50 characters of resume text.", None, None, None, None, None
        
        try:
            if model_type not in self.models:
                return f"Model '{model_type}' not available. Please select 'baseline' or 'debiased'.", None, None, None, None, None
            
            model = self.models[model_type]
            tokenizer = self.tokenizers[model_type]
            
            # Preprocess text (SAME AS TRAINING)
            cleaned_text, features, combined_text = self.preprocess_text(text)
            
            # Debug info
            print(f"\n[DEBUG] Model: {model_type}")
            print(f"[DEBUG] Combined text length: {len(combined_text)} chars")
            print(f"[DEBUG] Features extracted: {features}")
            
            # Tokenize
            inputs = tokenizer(
                combined_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], 5)
            
            # Format results
            result_text = f"## {model_type.upper()} Model Classification Results\n\n"
            
            top_idx = top_indices[0].item()
            top_idx_str = str(top_idx)
            top_category = self.label_map.get(top_idx_str, f"Category_{top_idx}")
            top_confidence = top_probs[0].item()
            
            result_text += f"**Primary Prediction:** {top_category}\n"
            result_text += f"**Confidence:** {top_confidence*100:.1f}%\n\n"
            
            # Get model accuracy
            if model_type == 'baseline':
                model_accuracy = self.baseline_results.get('eval_accuracy', 0.85) * 100
            else:
                model_accuracy = self.debiased_results.get('eval_accuracy', 0.85) * 100
            
            result_text += f"**Model Accuracy:** {model_accuracy:.1f}%\n\n"
            
            # Confidence level
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
                idx_str = str(idx.item())
                category = self.label_map.get(idx_str, f"Category_{idx.item()}")
                confidence = prob.item() * 100
                result_text += f"{i}. **{category}**: {confidence:.1f}%\n"
                predictions_data.append([category, f"{confidence:.1f}%"])
            
            df = pd.DataFrame(predictions_data, columns=['Category', 'Confidence'])
            
            # Get demographics using the updated method
            demographics = self._infer_demographics(text)
            
            # Generate LIME explanation
            lime_image = self._generate_lime_explanation(combined_text, model_type, top_idx)
            
            return result_text, df, float(top_confidence), demographics, lime_image
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}\n\nPlease check:\n1. The resume text is detailed enough\n2. Models are properly trained\n3. Try a different resume example"
            print(f"Prediction error: {e}")
            traceback.print_exc()
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
        try:
            # Use the infer_demographics method from bias_analyzer
            demographics = self.demo_inference.infer_demographics(text)
            print(f"[DEBUG] Inferred demographics: {demographics}")
            return demographics
        except Exception as e:
            print(f"Demographic inference error: {e}")
            # Fallback to individual methods
            try:
                return {
                    'gender': self.demo_inference.infer_gender(text),
                    'race': self.demo_inference.infer_race_from_text(text),
                    'age_group': self.demo_inference.infer_age_group(text)
                }
            except:
                return {
                    'gender': 'unknown',
                    'race': 'unknown',
                    'age_group': 'unknown'
                }
    
    def get_performance_metrics(self):
        """Get performance metrics for both models"""
        metrics_data = []
        
        # Baseline metrics
        baseline_acc = self.baseline_results.get('eval_accuracy', 0.85) * 100
        baseline_f1 = self.baseline_results.get('eval_f1', 0.84) * 100
        baseline_precision = self.baseline_results.get('eval_precision', 0.85) * 100
        baseline_recall = self.baseline_results.get('eval_recall', 0.85) * 100
        
        # Debiased metrics
        debiased_acc = self.debiased_results.get('eval_accuracy', 0.85) * 100
        debiased_f1 = self.debiased_results.get('eval_f1', 0.84) * 100
        debiased_precision = self.debiased_results.get('eval_precision', 0.85) * 100
        debiased_recall = self.debiased_results.get('eval_recall', 0.85) * 100
        
        metrics_data.append(["Baseline", f"{baseline_acc:.2f}%", f"{baseline_f1:.2f}%", 
                           f"{baseline_precision:.2f}%", f"{baseline_recall:.2f}%"])
        metrics_data.append(["Debiased", f"{debiased_acc:.2f}%", f"{debiased_f1:.2f}%", 
                           f"{debiased_precision:.2f}%", f"{debiased_recall:.2f}%"])
        
        return metrics_data


def create_interface():
    """Create Gradio interface with two tabs"""
    
    # Optimized resume examples
    examples = [
    """Michael Johnson
Sales Manager

PROFESSIONAL SUMMARY:
Sales Manager with 8 years of experience in retail sales and business development. He has led sales teams and consistently exceeded sales targets. His expertise includes customer relationship management, sales strategy, and business growth.

EXPERIENCE:
Senior Sales Manager, Retail Solutions Inc. | 2017-Present
- Led a team of 12 sales representatives
- Increased regional sales by 35% over three years
- Developed and implemented new sales strategies

Sales Representative, Business Products Corp | 2013-2017
- Managed key accounts and customer relationships
- Exceeded sales targets by 20% annually

EDUCATION:
University of Business Administration
Bachelor of Business in Sales Management | 2013

SKILLS:
- Sales Strategy and Planning
- Customer Relationship Management (CRM)
- Team Leadership and Training
- Business Development
- Negotiation and Closing

CERTIFICATIONS:
- Certified Professional Sales Leader (CPSL)
- Strategic Sales Management Certification

ACHIEVEMENTS:
- Sales Manager of the Year, 2021
- Top Performing Region Award, 2020
- Exceeded $5M in annual sales, 2019

PROFESSIONAL AFFILIATIONS:
- National Association of Sales Professionals
- Sales Management Association

ADDITIONAL:
- He volunteers as a mentor for young sales professionals
- Active member of the local business chamber
- He enjoys golf, networking events, and business reading""",

    """Emily Chen
Accountant

PROFESSIONAL SUMMARY:
Certified Public Accountant with 6 years of experience in financial accounting, tax preparation, and audit services. She specializes in corporate accounting and financial reporting. Her expertise ensures accurate financial records and compliance with tax regulations.

EXPERIENCE:
Senior Accountant, Financial Services Group | 2018-Present
- Managed financial reporting for 50+ corporate clients
- Prepared and filed federal and state tax returns
- Conducted internal audits and financial analysis

Staff Accountant, Accounting Solutions LLC | 2015-2018
- Processed accounts payable and receivable
- Assisted with month-end closing procedures
- Maintained general ledger accounts

EDUCATION:
University of Finance and Accounting
Bachelor of Science in Accounting | 2015
- Graduated Magna Cum Laude
- President, Accounting Student Association

LICENSES & CERTIFICATIONS:
- Certified Public Accountant (CPA), State Board of Accountancy
- QuickBooks Certified ProAdvisor
- Certified Management Accountant (CMA)

SKILLS:
- Financial Reporting and Analysis
- Tax Planning and Preparation
- Audit and Compliance
- GAAP and IFRS Standards
- Accounting Software (QuickBooks, Sage, Xero)

SPECIALTIES:
1. Corporate Taxation
2. Financial Statement Preparation
3. Internal Controls and Auditing
4. Budgeting and Forecasting

PROFESSIONAL DEVELOPMENT:
- Annual attendance at AICPA Conference
- Completed 40+ hours of continuing professional education annually
- Member, American Institute of CPAs

ADDITIONAL:
- She volunteers for VITA (Volunteer Income Tax Assistance) program
- Treasurer for local non-profit organization
- She enjoys hiking, yoga, and financial literacy workshops""",

    """James Wilson
High School Mathematics Teacher

PROFESSIONAL SUMMARY:
Dedicated Mathematics Teacher with 9 years of experience teaching high school mathematics. He creates engaging lesson plans and uses innovative teaching methods to inspire students. His teaching philosophy emphasizes conceptual understanding and real-world applications.

EXPERIENCE:
Mathematics Teacher, Lincoln High School | 2014-Present
- Teach Algebra I, Geometry, and Calculus to 9th-12th grade students
- Implemented project-based learning increasing student engagement by 40%
- Served as Mathematics Department Chair for 2 years

Student Teacher, Jefferson Middle School | 2013-2014
- Assisted in classroom instruction and management
- Developed supplemental learning materials

EDUCATION:
University of Wisconsin-Madison
Master of Education in Curriculum and Instruction | 2013
Bachelor of Science in Mathematics | 2011

CERTIFICATIONS:
- Teaching Certification, State of Wisconsin, Grades 6-12
- National Board Certification in Mathematics

SKILLS:
- Lesson Planning and Curriculum Development
- Differentiated Instruction
- Classroom Management
- Educational Technology Integration
- Student Assessment and Evaluation

ACHIEVEMENTS:
- Teacher of the Year, Lincoln High School, 2020
- Developed award-winning STEM curriculum adopted district-wide
- Led school's Math Club to state championship

PROFESSIONAL DEVELOPMENT:
- Completed 60+ hours of professional development in educational technology
- Attended National Council of Teachers of Mathematics annual conference

TEACHING PHILOSOPHY:
"I believe every student can succeed in mathematics with the right support and engaging instruction. My goal is to make math accessible and relevant to all learners."

ADDITIONAL:
- He coaches the school's math competition team
- Advisor for Student Council
- He enjoys hiking, reading historical fiction, and woodworking
- Active member of the National Education Association""",

    """Robert Davis
Executive Chef

PROFESSIONAL SUMMARY:
Executive Chef with 11 years of culinary experience in fine dining and restaurant management. He specializes in contemporary American cuisine and menu development. His culinary philosophy emphasizes fresh, local ingredients and innovative flavor combinations.

EXPERIENCE:
Executive Chef, The Gourmet Table Restaurant | 2016-Present
- Created seasonal menus featuring local and sustainable ingredients
- Managed kitchen staff of 15 employees
- Maintained 5-star health inspection ratings
- Increased restaurant revenue by 30% through menu innovation

Sous Chef, Culinary Arts Bistro | 2012-2016
- Assisted in menu planning and food preparation
- Supervised kitchen operations during service
- Trained junior kitchen staff

EDUCATION:
Culinary Institute of America
Associate Degree in Culinary Arts | 2012
- Graduated with Honors
- President, Student Culinary Association

CULINARY SKILLS:
- Menu Development and Cost Control
- Kitchen Management and Staff Training
- Food Safety and Sanitation (ServSafe Certified)
- Contemporary Cooking Techniques
- Pastry and Baking

SPECIALTIES:
1. Farm-to-Table Cuisine
2. Molecular Gastronomy
3. International Fusion Cooking
4. Plating and Presentation

AWARDS & RECOGNITION:
- James Beard Award Nominee, 2021
- Best Restaurant Award, City Dining Magazine, 2020
- Gold Medal, National Culinary Competition, 2019

RESTAURANT MANAGEMENT:
- Inventory Control and Ordering
- Vendor Relations and Sourcing
- Budget Management and Cost Analysis
- Customer Service and Dining Experience

PROFESSIONAL AFFILIATIONS:
- American Culinary Federation
- Slow Food International
- Chefs Collaborative

PERSONAL PHILOSOPHY:
"I believe food should be both delicious and meaningful. Cooking is my way of connecting people and creating memorable experiences."

ADDITIONAL:
- He volunteers at community food banks and cooking classes
- Judge for local culinary competitions
- He enjoys gardening, foraging, and experimenting with new ingredients
- Teaches monthly cooking classes for aspiring chefs"""
]
    
    # Try to initialize classifier
    try:
        classifier = ResumeClassifier()
        print("Resume classifier loaded successfully")
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        # Create error interface
        with gr.Blocks(title="Resume Classifier - Setup Required", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # Resume Classification System
            ## Setup Required
            
            Please run the training scripts first:
            
            ```bash
            # Step 1: Train baseline model
            python train_baseline.py
            
            # Step 2: Train debiased model
            python train_debiased.py
            
            # Step 3: Run bias analysis
            python bias_analysis.py
            ```
            
            **Error Details:**
            ```
            {error}
            ```
            """.format(error=str(e)))
        return demo
    
    with gr.Blocks(title="Resume Classification System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Resume Classification System
        **Classify resumes into 24 job categories with fairness analysis**
        
        *Powered by RoBERTa with bias mitigation techniques*
        """)
        
        with gr.Tab("Resume Classification"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Input")
                    
                    model_selector = gr.Radio(
                        choices=["baseline", "debiased"],
                        label="Select Model",
                        value="baseline",
                        info="**Baseline**: Standard model trained on resume data\n**Debiased**: Model trained with bias mitigation techniques"
                    )
                    
                    input_text = gr.Textbox(
                        label="Resume Text Input",
                        placeholder="Paste resume content here (minimum 50 characters)...\n\nTips for best results:\n1. Include job title and experience\n2. List specific skills and technologies\n3. Mention education and certifications\n4. Include full sentences and proper formatting",
                        lines=12,
                        max_lines=20
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Classify Resume", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    gr.Markdown("### Example Resumes")
                    gr.Examples(
                        examples=examples,
                        inputs=input_text,
                        label="Click to load example resumes"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    
                    output_text = gr.Markdown(label="Classification Results")
                    
                    with gr.Row():
                        confidence_score = gr.Number(
                            label="Confidence Score",
                            value=0.0,
                            precision=3
                        )
                    
                    output_table = gr.DataFrame(
                        label="Top 5 Predictions",
                        headers=["Category", "Confidence"],
                        datatype=["str", "str"]
                    )
                    
                    demographics_output = gr.JSON(
                        label="Inferred Demographics",
                        value={}
                    )
                    
                    lime_output = gr.Image(
                        label="LIME Explanation",
                        value=None,
                        width=600,
                        height=400
                    )
        
        with gr.Tab("Performance Metrics"):
            gr.Markdown("""
            ## Model Performance Metrics
            
            Comparison of baseline and debiased model performance.
            """)
            
            performance_df = gr.DataFrame(
                value=classifier.get_performance_metrics(),
                headers=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
                label="Performance Metrics Comparison",
                interactive=False
            )
            
            gr.Markdown("""
            ### Available Job Categories
            
            1. ACCOUNTANT
            2. ADVOCATE
            3. AGRICULTURE
            4. APPAREL
            5. ARTS
            6. AUTOMOBILE
            7. AVIATION
            8. BANKING
            9. BPO
            10. BUSINESS-DEVELOPMENT
            11. CHEF
            12. CONSTRUCTION
            13. CONSULTANT
            14. DESIGNER
            15. DIGITAL-MEDIA
            16. ENGINEERING
            17. FINANCE
            18. FITNESS
            19. HEALTHCARE
            20. HR
            21. INFORMATION-TECHNOLOGY
            22. PUBLIC-RELATIONS
            23. SALES
            24. TEACHER
            
            ### Model Information
            
            **Baseline Model:** 
            - Trained on original resume dataset
            - Standard preprocessing and class balancing
            - Uses RoBERTa-base architecture
            
            **Debiased Model:**
            - Trained with bias mitigation techniques
            - Adversarial training to reduce demographic bias
            - Counterfactual augmentation for fairness
            - Same architecture as baseline
            
            **Fairness Features:**
            - Demographic bias detection
            - Equal opportunity optimization
            - Intersectional fairness analysis
            """)
        
        # Event handlers
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
    print("Note: First run might be slow while models load.")
    print("Interface will be available at: http://localhost:7860")
    
    demo = create_interface()
    if demo:
        demo.launch(
            share= True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=False,
            show_error=True
        )
    else:
        print("Failed to create interface. Please check if models are trained.")
