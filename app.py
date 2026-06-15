"""
ML Pipeline Guide — LLM-Assisted End-to-End Machine Learning Pipeline Tool

Run with: streamlit run app.py
"""

import re
import streamlit as st
import json
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# PHASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

PHASES = {
    1: {
        "title": "Phase 1 — Problem Definition",
        "subtitle": "Research Framing",
        "icon": "🎯",
        "description": "Translate the business/research problem into a precise ML task.",
        "step": "Step 1: Define the Problem Statement",
        "inputs": [
            {
                "key": "objective",
                "label": "Objective",
                "type": "textarea",
                "help": "What is the overall goal of this project?",
                "placeholder": "e.g., Predict 30-day hospital readmission to enable early clinical intervention",
                "required": True,
                "default": "Develop a multi-class NLP model to automatically classify free-text pharmacy intervention documents by intervention type, enabling automated quality audit and workload analysis in the hospital pharmacy.",
            },
            {
                "key": "target_variable",
                "label": "Target Variable",
                "type": "text",
                "help": "What specific outcome are you predicting?",
                "placeholder": "e.g., Binary readmission within 30 days (yes/no)",
                "required": True,
                "default": "Intervention category (5 classes: Drug-Drug Interaction, Dose Adjustment, Therapeutic Substitution, Allergy Alert, Other)",
            },
            {
                "key": "prediction_window",
                "label": "Prediction Window",
                "type": "text",
                "help": "When are predictions made, and for what time horizon?",
                "placeholder": "e.g., At discharge, predict 30-day readmission",
                "required": True,
                "default": "At time of document submission — classify the intervention document into one of 5 categories immediately upon entry",
            },
            {
                "key": "unit_of_analysis",
                "label": "Unit of Analysis",
                "type": "text",
                "help": "What entity is being modelled?",
                "placeholder": "e.g., Patient-encounter pair",
                "required": True,
                "default": "Individual pharmacy intervention document",
            },
            {
                "key": "data_sources",
                "label": "Data Sources",
                "type": "textarea",
                "help": "List available data sources. Do NOT include actual data values.",
                "placeholder": "e.g., EHR system, lab results table, medication records, demographics",
                "required": True,
                "default": "Pharmacy intervention records system (free-text intervention documents, ~10,000 records), medication dispensing system (drug names, doses), pharmacist-assigned labels (historical ground truth), patient demographics (age, ward)",
            },
            {
                "key": "constraints",
                "label": "Constraints",
                "type": "textarea",
                "help": "Technical, clinical, regulatory, or operational constraints",
                "placeholder": "e.g., Model must be interpretable, real-time inference required, no external data",
                "required": False,
                "default": "Model must run on standard hospital server without GPU. Inference must complete within 2 seconds. Model predictions must be auditable — interpretability required. No patient data may leave the hospital network.",
            },
            {
                "key": "success_metrics",
                "label": "Success Metrics",
                "type": "textarea",
                "help": "How will success be defined and measured?",
                "placeholder": "e.g., AUROC > 0.80, sensitivity > 70% at 80% specificity",
                "required": True,
                "default": "Macro ROC-AUC > 0.85, macro F1 > 0.80, per-class sensitivity > 70%. Model must outperform a TF-IDF + Logistic Regression baseline.",
            },
            {
                "key": "deployment_goal",
                "label": "Deployment / Research Goal",
                "type": "textarea",
                "help": "What is the intended end use of this model?",
                "placeholder": "e.g., Production deployment in clinical workflow, or research publication in JAMIA",
                "required": True,
                "default": "Production deployment in the hospital pharmacy workflow to assist pharmacists with intervention categorisation and monthly quality reporting. Secondary goal: research publication.",
            },
        ],
        "prompt_template": """You are a senior AI scientist and healthcare ML researcher.

Help me refine the following ML problem and generate the output as a structured markdown document. This document will be used as input for subsequent pipeline steps.

Please address each of the following in detail:
1. Precise ML task definition
2. Prediction target definition
3. Label leakage risks
4. Suitable evaluation metrics
5. Potential biases
6. Clinical/business usefulness
7. Recommended validation strategy
8. Suggested study design for publication-quality work

## Problem Details

**Objective:**
{objective}

**Target Variable:**
{target_variable}

**Prediction Window:**
{prediction_window}

**Unit of Analysis:**
{unit_of_analysis}

**Data Sources:**
{data_sources}

**Constraints:**
{constraints}

**Success Metrics:**
{success_metrics}

**Deployment / Research Goal:**
{deployment_goal}

---

## Pipeline Metadata
At the end of your response, include the following block exactly as shown (single line per field):

**Pipeline.Domain:** [primary application domain, e.g., Healthcare / Clinical NLP]
**Pipeline.TaskType:** [ML task type, e.g., Multi-class text classification]
""",
        "output_filename": "phase1_problem_definition.md",
    },

    2: {
        "title": "Phase 2 — Literature Review",
        "subtitle": "Model Shortlisting",
        "icon": "📚",
        "description": "Research appropriate model families to avoid defaulting to standard choices.",
        "step": "Step 2: Model Recommendation via Literature Review",
        "optional": True,
        "primary_input": "problem_definition",
        "inputs": [
            {
                "key": "problem_statement",
                "label": "Problem Statement",
                "type": "textarea",
                "help": "Concise ML task definition used for literature search — auto-filled from Phase 1 output",
                "placeholder": "e.g., This is a multi-class text classification problem where free-text pharmacy intervention notes are classified into intervention categories...",
                "required": True,
            },
            {
                "key": "problem_definition",
                "label": "Problem Definition (Phase 1 Output)",
                "type": "file_or_text",
                "help": "Upload the Phase 1 output file, or paste/type the content directly",
                "placeholder": "Paste Phase 1 output here...",
                "required": True,
            },
            {
                "key": "domain",
                "label": "Domain",
                "type": "text",
                "help": "Primary application domain",
                "placeholder": "e.g., Healthcare, Clinical NLP, Tabular clinical data",
                "required": True,
            },
            {
                "key": "task_type",
                "label": "Task Type",
                "type": "text",
                "help": "Type of ML task",
                "placeholder": "e.g., Multi-class text classification, Binary tabular classification, Regression",
                "required": True,
            },
            {
                "key": "novelty_aspects",
                "label": "Novelty Aspects (Optional)",
                "type": "textarea",
                "help": "Specific novelty angles to explore for publication",
                "placeholder": "e.g., Federated learning, privacy-preserving methods, novel feature engineering",
                "required": False,
            },
            {
                "key": "sources_of_literature",
                "label": "Sources of Literature (Optional)",
                "type": "textarea",
                "help": "List specific journals, conferences, databases, or known papers to focus on",
                "placeholder": "e.g., PubMed, JAMIA, NeurIPS, Nature Digital Medicine; Smith et al. 2023 on transformer models for EHR",
                "required": False,
            },
        ],
        "prompt_template": """You are an expert ML researcher with deep knowledge of applied machine learning in {domain}.

Based on the following ML problem, conduct a thorough literature review and generate the output as a structured markdown document.

## Task Type: {task_type}

Please address the following:
1. Recommend promising model families with rationale
2. Explain WHY each model is suitable for this specific problem
3. Compare pros/cons of each approach
4. Identify SOTA approaches from recent literature
5. Suggest appropriate baseline models
6. Suggest interpretable models suitable for clinical/business settings
7. Suggest publication-worthy advanced models
8. Rank models by expected performance
9. Highlight key risks: overfitting, data leakage, class imbalance
10. Recommend evaluation metrics aligned with the problem requirements
11. Ideas for novelty contributions in this space
12. Common evaluation benchmarks used in this domain

**Novelty Aspects to Consider:**
{novelty_aspects}

**Preferred Literature Sources:**
{sources_of_literature}

## Problem Statement:
{problem_statement}
""",
        "output_filename": "phase2_literature_review.md",
    },

    3: {
        "title": "Phase 3 — EDA Planning",
        "subtitle": "Data Understanding",
        "icon": "🔍",
        "description": "Generate a comprehensive EDA notebook to understand the dataset structure and quality.",
        "step": "Step 3: Plan Exploratory Data Analysis",
        "optional": True,
        "primary_input": "phase1_output",
        "inputs": [
            {
                "key": "problem_statement",
                "label": "Problem Statement",
                "type": "textarea",
                "help": "Brief description of the ML problem",
                "placeholder": "e.g., Develop a multi-class model where input is free-text pharmacy intervention documents and output is a single label",
                "required": True,
            },
            {
                "key": "phase1_output",
                "label": "Problem Definition (Phase 1 Output)",
                "type": "file_or_text",
                "help": "Optional — upload or paste the Phase 1 output to prefill Problem Statement and Dataset Description below, if available",
                "placeholder": "Paste Phase 1 output here...",
                "required": False,
            },
            {
                "key": "dataset_description",
                "label": "Dataset Description",
                "type": "textarea",
                "help": "Describe the dataset structure. Do NOT include actual data values.",
                "placeholder": "e.g., ~10,000 rows. Columns: patient_id (int), intervention_text (string), label (string with 5 classes), date (datetime). ~15% missing in lab columns.",
                "required": True,
            },
            {
                "key": "target_column",
                "label": "Target Column Name",
                "type": "text",
                "help": "Name of the target/label column",
                "placeholder": "e.g., Label",
                "required": True,
            },
            {
                "key": "data_folder",
                "label": "Data Folder Path",
                "type": "text",
                "help": "Local path to the dataset file(s)",
                "placeholder": "e.g., ./data/",
                "required": True,
            },
            {
                "key": "output_folder",
                "label": "Output Folder Path",
                "type": "text",
                "help": "Where to save the EDA notebook and outputs",
                "placeholder": "e.g., ./outputs/eda/",
                "required": True,
            },
            {
                "key": "special_considerations",
                "label": "Special Considerations",
                "type": "textarea",
                "help": "Domain-specific EDA concerns to investigate",
                "placeholder": "e.g., Data is ordered by date — check temporal leakage; significant class imbalance expected; free-text needs NLP preprocessing",
                "required": False,
            },
        ],
        "prompt_template": """You are a senior data scientist with expertise in healthcare data analysis.

Given the dataset described below, generate a complete Jupyter Notebook (.ipynb) to conduct comprehensive EDA. The output of this notebook will be used to derive data insights in the next pipeline step.

## Problem Statement:
{problem_statement}

## Dataset Description:
{dataset_description}

## Target Column:
{target_column}

## Data Location:
{data_folder}

## Output Location:
{output_folder}

## Special Considerations:
{special_considerations}

---

The notebook must cover:
1. Full EDA checklist with progress tracking cells
2. Data quality checks (missing values, duplicates, schema validation, data type verification)
3. Missing value analysis (patterns, visualisations, MCAR/MAR/MNAR assessment)
4. Outlier detection strategy (statistical and visual methods)
5. Target leakage checks (temporal and informational)
6. Class imbalance analysis with visualisations
7. Temporal leakage checks (if time-ordered data)
8. Correlation analysis (numerical and categorical)
9. Feature importance exploration (mutual information, chi-square)
10. Visualisation suite (distributions, relationships, heatmaps, word clouds if text)
11. Bias and fairness checks (demographic parity, representation gaps)
12. Summary of expected clinical/business insights

**Important:** Generate complete, runnable Python code. Include markdown cells explaining the purpose of each section. Assume the data is sensitive — do not hardcode actual sample values. Save all plots and summary statistics to the output folder.
""",
        "output_filename": "phase3_eda_notebook_request.md",
    },

    4: {
        "title": "Phase 4 — EDA Insights",
        "subtitle": "Deriving Modelling Guidance",
        "icon": "💡",
        "description": "Translate EDA findings into structured, actionable modelling guidance.",
        "step": "Step 4: Derive Insights from EDA Results",
        "optional": True,
        "inputs": [
            {
                "key": "problem_statement",
                "label": "Problem Statement",
                "type": "textarea",
                "help": "Brief description of the ML problem",
                "placeholder": "e.g., Develop a multi-class model for pharmacy intervention classification",
                "required": True,
                "hidden": True,
            },
            {
                "key": "target_column",
                "label": "Ground Truth / Target Column",
                "type": "text",
                "help": "Column name of the target variable",
                "placeholder": "e.g., Label",
                "required": True,
            },
            {
                "key": "eda_results",
                "label": "EDA Results / Analysis Output",
                "type": "file_or_text",
                "help": "Upload the EDA output file, or paste/type the content directly. NOT raw data rows.",
                "placeholder": "Paste EDA analysis results here (e.g., summary statistics, key findings, outputs from EDA notebook markdown cells)...",
                "required": True,
            },
        ],
        "prompt_template": """You are a senior data scientist assisting with the transition from exploratory data analysis (EDA) to experimental design and predictive modelling.

Based strictly on the EDA results provided below, generate a structured insights report as a markdown document to guide the next stage of experiment and modelling planning.

**Important constraints:**
- Do NOT run any code
- Base your insights ONLY on what is provided in the EDA results
- The underlying dataset is sensitive and is not available here
- Be critical and specific — avoid vague statements like "the data looks good"
- If something is unclear, explicitly state what additional analysis would clarify it

The report must contain:

### 1. Critical Data Characteristics
- Missingness patterns and recommended imputation strategies
- Class imbalance severity and implications for model training
- Outlier presence and recommended handling approach
- Scaling and normalisation requirements

### 2. Candidate Drivers
- Features most strongly associated with the target variable
- Non-linear relationships or interaction hints
- Text/NLP feature insights (if applicable)

### 3. Identified Risks & Limitations
- Data leakage suspicion (temporal or informational)
- Low-variance or near-duplicate features to remove
- Confounding signals that may mislead the model
- Data quality concerns requiring remediation

### 4. Testable Hypotheses
2–3 specific, actionable hypotheses for A/B experiments or ablation studies.

### 5. Modelling Implications
- Recommended model families given the data characteristics
- Feature engineering ideas
- Recommended validation strategy
- Class imbalance handling strategy

## Problem Statement:
{problem_statement}

## Ground Truth / Target Column:
{target_column}

## EDA Results:
{eda_results}
""",
        "output_filename": "phase4_eda_insights.md",
    },

    5: {
        "title": "Phase 5 — Preprocessing",
        "subtitle": "Feature Engineering Pipeline",
        "icon": "⚙️",
        "description": "Generate a data preprocessing and feature engineering pipeline notebook.",
        "step": "Step 5: Build Preprocessing Pipeline",
        "inputs": [
            {
                "key": "problem_statement",
                "label": "Problem Statement",
                "type": "textarea",
                "help": "Brief description of the ML problem",
                "placeholder": "e.g., Develop a multi-class model for pharmacy intervention classification",
                "required": True,
                "hidden": True,
            },
            {
                "key": "data_description",
                "label": "Data Description",
                "type": "textarea",
                "help": "Describe data types, columns, and structure. No actual data values.",
                "placeholder": "e.g., Text column 'intervention_text', categorical 'department' (5 categories), numerical 'age' and 'weight', target 'label'",
                "required": True,
            },
            {
                "key": "eda_insights",
                "label": "EDA Insights (Phase 4 Output)",
                "type": "file_or_text",
                "help": "Upload the Phase 4 output file, or paste/type the content directly. Optional if Phase 4 was skipped.",
                "placeholder": "Paste EDA insights here...",
                "required": False,
            },
            {
                "key": "output_folder",
                "label": "Output Folder Path",
                "type": "text",
                "help": "Where to save processed data and pipeline artefacts",
                "placeholder": "e.g., ./outputs/preprocessing/",
                "required": True,
            },
        ],
        "prompt_template": """You are a senior data scientist developing a data preprocessing and feature engineering pipeline.

## Problem Statement:
{problem_statement}

## Data Description:
{data_description}

## Output Folder:
{output_folder}

Based on the EDA insights below, generate a Jupyter Notebook (.ipynb) for data preprocessing and feature engineering.

### Requirements:
- Handle missing values using imputation strategy justified by EDA findings
- Encode categorical variables (consider cardinality — use ordinal/label/target encoding or OHE as appropriate)
- Scale numerical features (justify choice: StandardScaler vs MinMaxScaler vs RobustScaler based on EDA)
- Handle text features if present (tokenisation, TF-IDF, sentence embeddings — as appropriate)
- Address outliers where necessary
- Split data into stratified train, validation, and test sets; store all splits as files
- Wrap all transformations in a sklearn Pipeline or ColumnTransformer so they can be applied identically to train and test
- Include markdown cells explaining each preprocessing decision

### Outputs to Save:
- Preprocessed train/val/test CSV/parquet files
- Fitted pipeline artefacts (joblib/pickle for inference)
- Feature engineering summary report

## EDA Insights:
{eda_insights}
""",
        "output_filename": "phase5_preprocessing_plan.md",
    },

    6: {
        "title": "Phase 6 — Experimental Design",
        "subtitle": "Modelling Strategy",
        "icon": "🧪",
        "description": "Design a rigorous ML experimentation and modelling strategy aligned with literature and data findings.",
        "step": "Step 6: Generate Modelling Experiment Plan",
        "inputs": [
            {
                "key": "literature_review",
                "label": "Literature Review (Phase 2 Output)",
                "type": "file_or_text",
                "help": "Upload the Phase 2 output file, or paste/type the content directly. Optional if Phase 2 was skipped.",
                "placeholder": "Paste literature review here...",
                "required": False,
            },
            {
                "key": "eda_insights",
                "label": "EDA Insights (Phase 4 Output)",
                "type": "file_or_text",
                "help": "Upload the Phase 4 output file, or paste/type the content directly. Optional if Phase 4 was skipped.",
                "placeholder": "Paste EDA insights here...",
                "required": False,
            },
            {
                "key": "evaluation_metrics",
                "label": "Evaluation Metrics Required by Stakeholders",
                "type": "textarea",
                "help": "List all required evaluation metrics",
                "placeholder": "e.g., macro ROC-AUC, micro ROC-AUC, maximum F1",
                "required": True,
            },
            {
                "key": "implementation_constraints",
                "label": "Implementation Constraints",
                "type": "textarea",
                "help": "Computational, time, or resource constraints",
                "placeholder": "e.g., Must run on standard laptop without GPU; lightweight models preferred for deployment",
                "required": False,
            },
        ],
        "prompt_template": """You are a principal machine learning scientist and healthcare AI researcher.

Design a rigorous machine learning experimentation and modelling strategy. Save it as a structured markdown document. This will serve as the experiment plan for code generation in Phase 7.

**Note:** Train, validation, and test sets have already been split.

Your design must:
- Align with the literature review findings
- Adapt to actual dataset characteristics from EDA
- Be realistic and implementable
- Maximise robustness and reproducibility
- Consider deployment and resource constraints

## Evaluation Metrics Required by Stakeholders:
{evaluation_metrics}

## Implementation Constraints:
{implementation_constraints}

---

The experiment strategy document should include:

### 1. Recommended Model Families
For each model family provide:
- Rationale and suitability for this specific problem
- Expected strengths
- Expected weaknesses
- Deployment considerations

### 2. Prioritised Model List
With clear justification for priority ordering.

### 3. Iterative Enhancement Method
Strategy for progressively improving model performance across experiments.

### 4. Validation Framework
- Cross-validation strategy
- Holdout strategy
- Preventing data leakage in validation

### 5. Explainability Approach
Tools and methods for model interpretability (SHAP, LIME, attention weights, etc.).

### 6. Experiment Tracking Strategy
How to log, compare, and reproduce experiments.

### 7. Risk Mitigation
- Overfitting prevention strategies
- Data leakage prevention
- Class imbalance handling

---

## Literature Review Findings:
{literature_review}

## EDA Insights:
{eda_insights}
""",
        "output_filename": "phase6_experiment_plan.md",
    },

    7: {
        "title": "Phase 7 — Training Pipeline",
        "subtitle": "Code Generation",
        "icon": "🚀",
        "description": "Generate a complete, modular, end-to-end ML pipeline as a Jupyter Notebook.",
        "step": "Step 7: Generate Modular ML Pipeline Code",
        "inputs": [
            {
                "key": "problem_statement",
                "label": "Problem Statement",
                "type": "textarea",
                "help": "Brief description of the ML problem",
                "placeholder": "e.g., Develop a multi-class model where input is free-text pharmacy intervention documents and output is a single label",
                "required": True,
                "hidden": True,
            },
            {
                "key": "experiment_plan",
                "label": "Experiment Plan (Phase 6 Output)",
                "type": "file_or_text",
                "help": "Upload the Phase 6 output file, or paste/type the content directly",
                "placeholder": "Paste experiment plan here...",
                "required": True,
            },
            {
                "key": "data_folder",
                "label": "Data Folder Path",
                "type": "text",
                "help": "Folder containing train/val/test splits",
                "placeholder": "e.g., ./data/splits/",
                "required": True,
            },
            {
                "key": "input_column",
                "label": "Input Feature Column",
                "type": "text",
                "help": "Column name of the model input feature",
                "placeholder": "e.g., Intervention Document",
                "required": True,
            },
            {
                "key": "target_column",
                "label": "Target / Label Column",
                "type": "text",
                "help": "Column name of the target variable",
                "placeholder": "e.g., Label",
                "required": True,
            },
            {
                "key": "evaluation_metrics",
                "label": "Evaluation Metrics",
                "type": "text",
                "help": "Comma-separated list of required metrics",
                "placeholder": "e.g., macro ROC-AUC, micro ROC-AUC, max F1",
                "required": True,
            },
            {
                "key": "additional_requirements",
                "label": "Additional Requirements",
                "type": "textarea",
                "help": "Specific requirements for the notebook",
                "placeholder": "e.g., Include SHAP explanations, export tracking table to Excel, support GPU if available",
                "required": False,
            },
        ],
        "prompt_template": """You are a senior machine learning engineer and AI researcher.

Generate a complete end-to-end Python experimentation and modelling pipeline as a Jupyter Notebook (.ipynb).

The notebook must be interactive, modular, and visualisation-focused so users can directly inspect:
- Model structures and configurations
- Experiment outputs and training progress
- Evaluation metrics across all experiments
- Experiment comparison results and rankings
- Error analysis and misclassification patterns
- Explainability outputs

Only create separate .py helper files for genuinely reusable utilities. The notebook is the primary artefact.

## Problem Statement:
{problem_statement}

## Dataset Information:
- Data folder: `{data_folder}`
- Input column: `{input_column}`
- Target/label column: `{target_column}`
- Train, validation, and test splits already exist in the data folder

## Evaluation Metrics:
{evaluation_metrics}

## Additional Requirements:
{additional_requirements}

---

## Required Notebook Sections:

### Section 1: Setup & Configuration
- All imports and reproducibility seeds
- A configuration dictionary at the top for easy parameter changes

### Section 2: Data Loading & Validation
- Load train/val/test datasets
- Display shapes, data types, class distributions
- Train/val/test consistency checks
- Sample inspection

### Section 3: Preprocessing Pipeline
- Encoding or tokenising strategy with justification in markdown

### Section 4: Multi-Model Experimentation
Implement all models from the experiment plan (baselines + advanced models).
For every experiment, display:
- Model structure/configuration (readable summary)
- Training progress (loss/metric curves where applicable)
- Validation performance
- Hyperparameter tuning results

### Section 5: Experiment Tracking Table
Generate a comprehensive pandas DataFrame with columns:
`experiment_id | model_name | preprocessing_config | hyperparameters | macro_roc_auc_train | macro_roc_auc_val | macro_roc_auc_test | micro_roc_auc_train | micro_roc_auc_val | micro_roc_auc_test | max_f1_train | max_f1_val | max_f1_test | training_time | inference_time | notes`

- Display table with ranking in the notebook
- Export to Excel

### Section 6: Evaluation & Visualisation
- ROC curves (per class + macro/micro average)
- Confusion matrices
- Train vs validation vs test performance comparison
- Threshold optimisation plots
- Calibration plots
- Feature importance / SHAP plots

### Section 7: Error Analysis & Improvement Diagnostics
- False positives and false negatives analysis
- Hardest samples (lowest confidence correct predictions)
- Misclassification patterns by class
- Feature stability analysis
- Specific recommendations for next experiments, feature improvements, and tuning

### Section 8: Final Model Selection
- Best model identification with explicit selection criteria
- Selection rationale
- Strengths and weaknesses
- Deployment considerations and recommended next steps

---

The notebook must:
- Use rich markdown cells between sections with clear explanations
- Be easy to rerun cell by cell
- Avoid code duplication — use helper functions
- Support adding new experiments by duplicating an experiment cell block
- Display all outputs clearly inline
- Generate publication-quality visualisations (proper labels, titles, legends, figure sizes)

---

## Experiment & Modelling Plan:
{experiment_plan}
""",
        "output_filename": "phase7_training_pipeline_request.md",
    },
}

# ═══════════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════════

def _call_openai(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_gemini(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def _call_azure_openai(
    prompt: str, api_key: str, endpoint: str, deployment: str, temperature: float, max_tokens: int
) -> str:
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = AzureOpenAI(api_key=api_key, api_version="2024-02-01", azure_endpoint=endpoint)
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_llm(prompt: str, provider: str, config: dict) -> str:
    if provider == "OpenAI":
        return _call_openai(
            prompt, config["api_key"], config["model"],
            config["temperature"], config["max_tokens"]
        )
    elif provider == "Anthropic (Claude)":
        return _call_anthropic(
            prompt, config["api_key"], config["model"],
            config["temperature"], config["max_tokens"]
        )
    elif provider == "Google (Gemini)":
        return _call_gemini(
            prompt, config["api_key"], config["model"],
            config["temperature"], config["max_tokens"]
        )
    elif provider == "Azure OpenAI":
        return _call_azure_openai(
            prompt, config["api_key"], config.get("endpoint", ""),
            config.get("deployment", ""), config["temperature"], config["max_tokens"]
        )
    raise ValueError(f"Unknown provider: {provider}")


# ═══════════════════════════════════════════════════════════════════
# AUTOFILL MAP  — (source, src_phase_id, src_input_key)
# source="response" pulls the LLM output of src_phase_id
# source="input"    pulls a specific input field from src_phase_id
# ═══════════════════════════════════════════════════════════════════

def _parse_pipeline_field(response_text: str, field_name: str) -> str:
    """Extract a **Pipeline.<field>:** value from a Phase 1 LLM response."""
    match = re.search(
        rf'\*\*Pipeline\.{re.escape(field_name)}:\*\*\s*(.+?)(?:\n|$)',
        response_text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _parse_ml_task_first_paragraph(text: str) -> str:
    """Extract the first paragraph of the ML Task Definition section from a Phase 1 LLM response."""
    match = re.search(
        r'\n#{1,4}\s*\d*\.?\s*(?:Precise\s+)?ML\s+[Tt]ask\s+[Dd]efinition[^\n]*\n',
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    after_heading = text[match.end():]
    for para in re.split(r'\n\s*\n', after_heading):
        lines = [l for l in para.strip().splitlines() if not l.strip().startswith('#')]
        clean = ' '.join(lines).strip()
        if clean:
            return clean
    return ""


def _extract_from_primary_input(phase_id: int, content: str, saved_inputs: dict) -> list:
    """Parse structured fields from primary input content and pre-fill saved_inputs."""
    populated = []
    primary_key = PHASES[phase_id].get("primary_input")
    for field_key, sources in _AUTOFILL.get(phase_id, {}).items():
        if field_key == primary_key:
            continue
        if saved_inputs.get(field_key, "").strip():
            continue
        for source, _, src_field in sources:
            if source == "parse":
                val = _parse_pipeline_field(content, src_field)
                if val:
                    saved_inputs[field_key] = val
                    populated.append(field_key)
                    break

    # Phase 2 & 3: extract problem_statement from ML Task Definition section
    if phase_id in (2, 3) and not saved_inputs.get("problem_statement", "").strip():
        val = _parse_ml_task_first_paragraph(content)
        if val:
            saved_inputs["problem_statement"] = val
            populated.append("problem_statement")

    return populated


def _clear_autofilled_fields(phase_id: int, saved_inputs: dict):
    """Clear fields previously auto-extracted from the primary input, so a fresh load can repopulate them."""
    primary_key = PHASES[phase_id].get("primary_input")
    for field_key, sources in _AUTOFILL.get(phase_id, {}).items():
        if field_key == primary_key:
            continue
        if any(source == "parse" for source, _, _ in sources):
            saved_inputs.pop(field_key, None)
    if phase_id in (2, 3):
        saved_inputs.pop("problem_statement", None)


def _render_primary_input_section(phase_id: int, inp: dict, saved_inputs: dict):
    """Render Step 1 — load a previous phase output outside the form for immediate reactivity."""
    current_value = saved_inputs.get(inp["key"], "")

    if current_value.strip():
        char_count = len(current_value)
        with st.expander(
            f"✅ {inp['label']} loaded ({char_count:,} characters) — click to view or replace"
        ):
            st.markdown(
                current_value[:1500]
                + ("\n\n*[truncated — full content stored]*" if char_count > 1500 else "")
            )
            if st.button("🔄 Clear & Re-load", key=f"primary_clear_{phase_id}"):
                saved_inputs[inp["key"]] = ""
                _clear_autofilled_fields(phase_id, saved_inputs)
                st.rerun()
    else:
        tab_paste, tab_upload = st.tabs(["✏️ Paste Content", "📎 Upload File"])

        with tab_paste:
            pasted = st.text_area(
                "Paste content",
                height=200,
                placeholder=inp.get("placeholder", ""),
                label_visibility="collapsed",
                key=f"primary_paste_{phase_id}",
            )
            if st.button("Load & Extract →", key=f"primary_load_{phase_id}", type="primary"):
                if pasted.strip():
                    saved_inputs[inp["key"]] = pasted
                    populated = _extract_from_primary_input(phase_id, pasted, saved_inputs)
                    if populated:
                        st.session_state[f"_extracted_{phase_id}"] = populated
                    st.rerun()
                else:
                    st.warning("Please paste content before loading.")

        with tab_upload:
            st.caption("Accepted formats: `.md`, `.txt`")
            uploaded = st.file_uploader(
                f"Upload {inp['label']}",
                type=["md", "txt"],
                key=f"primary_upload_{phase_id}",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                try:
                    content = uploaded.read().decode("utf-8")
                    saved_inputs[inp["key"]] = content
                    populated = _extract_from_primary_input(phase_id, content, saved_inputs)
                    if populated:
                        st.session_state[f"_extracted_{phase_id}"] = populated
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")

    extracted = st.session_state.pop(f"_extracted_{phase_id}", None)
    if extracted:
        labels = {i["key"]: i["label"] for i in PHASES[phase_id]["inputs"]}
        names = [labels.get(k, k) for k in extracted]
        st.success(f"Auto-extracted: **{', '.join(names)}**")


_AUTOFILL = {
    2: {
        "problem_definition": [("response", 1, None)],
        "domain":             [("parse", 1, "Domain")],
        "task_type":          [("parse", 1, "TaskType")],
    },
    3: {
        "problem_statement":   [("input", 1, "objective")],
        "phase1_output":       [("response", 1, None)],
        "dataset_description": [("parse", 1, "DatasetDescription")],
    },
    4: {
        # Try Phase 3 input first; fall back to Phase 1 objective if Phase 3 was skipped
        "problem_statement": [("input", 3, "problem_statement"), ("input", 1, "objective")],
    },
    5: {
        "problem_statement": [("input", 3, "problem_statement"), ("input", 1, "objective")],
        "eda_insights":      [("response", 4, None)],
    },
    6: {
        "literature_review": [("response", 2, None)],
        "eda_insights":      [("response", 4, None)],
    },
    7: {
        "problem_statement": [("input", 3, "problem_statement"), ("input", 1, "objective")],
        "experiment_plan":   [("response", 6, None)],
    },
}


# ═══════════════════════════════════════════════════════════════════
# STATE HELPERS
# ═══════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_FILE = BASE_DIR / "pipeline_state.json"

_DEFAULT_LLM_CONFIG = {
    "provider": "OpenAI",
    "api_key": "",
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 4000,
    "endpoint": "",
    "deployment": "",
}


def init_session():
    defaults = {
        "current_phase": 0,
        "inputs": {},
        "responses": {},
        "prompts_shown": {},
        "mode": "Copy-Paste Mode",
        "llm_config": dict(_DEFAULT_LLM_CONFIG),
        "improved_prompts": {},
        "use_improved_prompt": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def save_session() -> str:
    state = {
        "current_phase": st.session_state.current_phase,
        "inputs": st.session_state.inputs,
        "responses": {
            k: {kk: vv for kk, vv in v.items() if kk != "api_key"}
            for k, v in st.session_state.responses.items()
        },
        "improved_prompts": st.session_state.improved_prompts,
        "use_improved_prompt": st.session_state.use_improved_prompt,
        "saved_at": datetime.now().isoformat(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(STATE_FILE.resolve())


def load_session(data: dict):
    st.session_state.current_phase = data.get("current_phase", 0)
    st.session_state.inputs = data.get("inputs", {})
    st.session_state.responses = data.get("responses", {})
    st.session_state.prompts_shown = {k: True for k in st.session_state.responses}
    st.session_state.improved_prompts = data.get("improved_prompts", {})
    st.session_state.use_improved_prompt = data.get("use_improved_prompt", {})


def generate_prompt(phase_id: int) -> str:
    phase = PHASES[phase_id]
    template = phase["prompt_template"]
    phase_inputs = st.session_state.inputs.get(str(phase_id), {})

    kwargs = {}
    for inp in phase["inputs"]:
        val = phase_inputs.get(inp["key"], "").strip()
        kwargs[inp["key"]] = val if val else ("Not specified" if not inp.get("required") else f"[{inp['label']} — not yet provided]")

    try:
        return template.format(**kwargs)
    except KeyError as exc:
        return f"[Prompt generation error — missing key: {exc}]"


def build_prompt_improvement_request(original_prompt: str) -> str:
    return f"""You are an expert prompt engineer. Improve the prompt between the markers below so it is clearer, more specific, and more likely to produce a high-quality, well-structured response from an LLM.

Requirements:
- Preserve all factual details, placeholders, and instructions exactly — do not remove or invent information
- Return ONLY the improved prompt text, with no commentary, preamble, or markdown code fences

---ORIGINAL PROMPT START---
{original_prompt}
---ORIGINAL PROMPT END---
"""


def save_output(content: str, filename: str, phase_id: int) -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    phase_dir = OUTPUT_DIR / f"phase{phase_id}"
    phase_dir.mkdir(exist_ok=True)
    filepath = phase_dir / filename
    filepath.write_text(content, encoding="utf-8")
    return str(filepath.resolve())


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 ML Pipeline Guide")
        st.caption("LLM-Assisted End-to-End ML Development")
        st.markdown("---")

        # Mode selection
        st.markdown("**Interaction Mode**")
        mode = st.radio(
            "Interaction Mode",
            ["Copy-Paste Mode", "API Mode"],
            index=0 if st.session_state.mode == "Copy-Paste Mode" else 1,
            label_visibility="collapsed",
            help="Copy-Paste Mode: prompts generated locally, submitted manually to any LLM service.\nAPI Mode: prompts sent directly to the provider API; only metadata is transmitted.",
            key="mode_radio",
        )
        st.session_state.mode = mode

        if mode == "Copy-Paste Mode":
            st.caption("Prompts are generated locally and submitted manually. No automatic data transmission.")
        else:
            st.caption("Prompts transmitted to the provider API. Only metadata descriptions are sent.")

        # API config
        if mode == "API Mode":
            st.markdown("---")
            st.markdown("**API Settings**")

            providers = ["OpenAI", "Anthropic (Claude)", "Google (Gemini)", "Azure OpenAI"]
            current_provider = st.session_state.llm_config.get("provider", "OpenAI")
            provider_idx = providers.index(current_provider) if current_provider in providers else 0
            provider = st.selectbox("Provider", providers, index=provider_idx)
            st.session_state.llm_config["provider"] = provider

            api_key = st.text_input(
                "API Key",
                value=st.session_state.llm_config.get("api_key", ""),
                type="password",
                help="Stored in session only — never written to disk.",
            )
            st.session_state.llm_config["api_key"] = api_key

            if provider == "OpenAI":
                model = st.selectbox(
                    "Model",
                    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-preview", "o1-mini"],
                )
                st.session_state.llm_config["model"] = model

            elif provider == "Anthropic (Claude)":
                model = st.selectbox(
                    "Model",
                    [
                        "claude-opus-4-8",
                        "claude-sonnet-4-6",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                    ],
                )
                st.session_state.llm_config["model"] = model

            elif provider == "Google (Gemini)":
                model = st.selectbox(
                    "Model",
                    [
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "gemini-2.0-flash",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                    ],
                )
                st.session_state.llm_config["model"] = model

            elif provider == "Azure OpenAI":
                endpoint = st.text_input(
                    "Azure Endpoint",
                    value=st.session_state.llm_config.get("endpoint", ""),
                    placeholder="https://your-resource.openai.azure.com/",
                )
                deployment = st.text_input(
                    "Deployment Name",
                    value=st.session_state.llm_config.get("deployment", ""),
                    placeholder="gpt-4o",
                )
                st.session_state.llm_config["endpoint"] = endpoint
                st.session_state.llm_config["deployment"] = deployment

            with st.expander("Advanced"):
                temp = st.slider(
                    "Temperature", 0.0, 1.0,
                    float(st.session_state.llm_config.get("temperature", 0.3)), 0.05,
                )
                max_tok = st.number_input(
                    "Max Tokens", 1000, 16000,
                    int(st.session_state.llm_config.get("max_tokens", 4000)), 500,
                )
                st.session_state.llm_config["temperature"] = temp
                st.session_state.llm_config["max_tokens"] = max_tok

        st.markdown("---")

        # Phase navigation
        st.markdown("**Navigation**")

        if st.button(
            "🏠 Overview",
            use_container_width=True,
            type="primary" if st.session_state.current_phase == 0 else "secondary",
        ):
            st.session_state.current_phase = 0
            st.rerun()

        for phase_id, phase in PHASES.items():
            done = str(phase_id) in st.session_state.responses
            status = "✅" if done else "○"
            phase_label = phase["title"].split("—")[0].strip()
            if phase.get("optional"):
                phase_label += " *(optional)*"
            label = f"{status} {phase['icon']} {phase_label}"
            is_current = phase_id == st.session_state.current_phase
            if st.button(
                label,
                key=f"nav_{phase_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                st.session_state.current_phase = phase_id
                st.rerun()

        st.markdown("---")

        # Session management
        st.markdown("**Session**")

        if st.button("💾 Save Progress", use_container_width=True):
            path = save_session()
            st.success("Saved!")
            st.caption(f"`{path}`")

        st.download_button(
            "⬇️ Export Session JSON",
            data=json.dumps(
                {
                    "current_phase": st.session_state.current_phase,
                    "inputs": st.session_state.inputs,
                    "responses": {
                        k: {kk: vv for kk, vv in v.items() if kk != "api_key"}
                        for k, v in st.session_state.responses.items()
                    },
                    "saved_at": datetime.now().isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            file_name="pipeline_session.json",
            mime="application/json",
            use_container_width=True,
        )

        uploaded = st.file_uploader("📂 Load Session JSON", type="json", label_visibility="visible")
        if uploaded is not None:
            try:
                load_session(json.load(uploaded))
                st.success("Session loaded!")
                st.rerun()
            except Exception as exc:
                st.error(f"Load failed: {exc}")


# ═══════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ═══════════════════════════════════════════════════════════════════

def render_overview():
    st.title("🔬 ML Pipeline Guide")
    st.markdown("### LLM-Assisted End-to-End Machine Learning Development")

    st.info(
        "**Designed for data scientists working with sensitive data that must remain within your environment.**\n\n"
        "This tool provides a structured, LLM-assisted workflow for end-to-end machine learning development. "
        "Choose an interaction mode based on your data governance requirements:\n\n"
        "- **Copy-Paste Mode** — Prompts are generated and remain on your local machine. "
        "You submit each prompt to an LLM service of your choice (e.g. Claude, ChatGPT, Gemini) and paste the response back. "
        "No data is transmitted automatically; the user retains full control at every step.\n"
        "- **API Mode** — Prompts are transmitted programmatically to the selected LLM provider via authenticated API. "
        "Only structured metadata descriptions are included in requests. "
        "Raw data values are never transmitted."
    )

    with st.expander("ℹ️ Why 7 phases? Understanding the recommended workflow"):
        st.markdown(
            "This pipeline is structured into 7 phases to reflect rigorous, publication-quality ML development practice. "
            "Each phase builds on the previous, reducing the risk of downstream errors such as data leakage, "
            "model overfitting, or misaligned evaluation.\n\n"
            "**Phases 2, 3 and 4 are marked optional** to accommodate scenarios where literature review or EDA "
            "has already been completed externally, or where time and resource constraints require a more "
            "streamlined approach. However, they are **strongly encouraged** for the following reasons:\n\n"
            "- **Phase 2 (Literature Review)** ensures the modelling approach is grounded in current research "
            "rather than defaulting to familiar models.\n"
            "- **Phase 3 (EDA Planning)** ensures you fully understand dataset quality, class balance, "
            "missingness, and potential leakage before committing to a modelling strategy.\n"
            "- **Phase 4 (EDA Insights)** translates raw EDA findings into actionable, structured guidance "
            "that directly informs preprocessing and model selection in later phases.\n\n"
            "Skipping these phases increases the risk of building models on poorly understood data. "
            "If Phase 2, 3 and 4 outputs are available from prior work, they can be uploaded directly in "
            "Phase 6 (Phase 2 and 4) and Phase 5 (Phase 4) without repeating the analysis."
        )

    st.markdown("---")

    # Progress summary
    completed = sum(1 for k in st.session_state.responses if k.isdigit())
    total = len(PHASES)
    st.markdown(f"**Overall Progress: {completed} / {total} phases complete**")
    st.progress(completed / total if total else 0)

    # Smart CTA — Start, Continue, or All Done
    next_phase = next((pid for pid in PHASES if str(pid) not in st.session_state.responses), None)
    if completed == 0:
        cta_label, cta_target = "▶  Start — Go to Phase 1", 1
    elif next_phase:
        cta_label = f"▶  Continue — {PHASES[next_phase]['icon']} {PHASES[next_phase]['title']}"
        cta_target = next_phase
    else:
        cta_label, cta_target = "🎉 All phases complete — Go to Phase 1", 1

    if st.button(cta_label, type="primary"):
        st.session_state.current_phase = cta_target
        st.rerun()

    st.markdown("---")

    # Compact phase status checklist (no Open buttons — sidebar handles navigation)
    st.subheader("Phase Status")
    for phase_id, phase in PHASES.items():
        done = str(phase_id) in st.session_state.responses
        has_inputs = bool(st.session_state.inputs.get(str(phase_id)))
        if done:
            status = "✅"
        elif has_inputs:
            status = "🔄"
        else:
            status = "○"
        optional = " &nbsp; *(optional)*" if phase.get("optional") else ""
        st.markdown(
            f"{status} &nbsp; **{phase['icon']} {phase['title']}**{optional}",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Quick Tips")
    st.markdown(
        "- **Carry context forward**: each phase's output is auto-filled into the next phase's inputs\n"
        "- **Outputs are saved** in the `outputs/` folder (created alongside `app.py`)\n"
        "- **Save your session** regularly — use the sidebar buttons to resume later\n"
        "- **Copy-Paste Mode is safest** for very sensitive data: only your text descriptions go to the LLM, never data values"
    )


# ═══════════════════════════════════════════════════════════════════
# PHASE PAGE
# ═══════════════════════════════════════════════════════════════════

def render_phase(phase_id: int):
    phase = PHASES[phase_id]
    done = str(phase_id) in st.session_state.responses

    # Header
    col_h, col_s = st.columns([5, 1])
    with col_h:
        st.title(f"{phase['icon']} {phase['title']}")
        st.caption(phase["subtitle"])
    with col_s:
        if done:
            st.success("✅ Done")
        else:
            st.warning("⏳ In progress")

    st.info(f"**{phase['step']}** — {phase['description']}")
    st.markdown("---")

    if str(phase_id) not in st.session_state.inputs:
        st.session_state.inputs[str(phase_id)] = {}

    saved_inputs = st.session_state.inputs[str(phase_id)]

    # Auto-fill empty fields from previous phase outputs (supports fallback source lists)
    for field_key, sources in _AUTOFILL.get(phase_id, {}).items():
        if saved_inputs.get(field_key, "").strip():
            continue
        for source, src_phase, src_field in sources:
            if source == "response":
                val = st.session_state.responses.get(str(src_phase), {}).get("content", "")
            elif source == "parse":
                resp = st.session_state.responses.get(str(src_phase), {}).get("content", "")
                val = _parse_pipeline_field(resp, src_field) if resp else ""
            else:
                val = st.session_state.inputs.get(str(src_phase), {}).get(src_field, "")
            if val:
                saved_inputs[field_key] = val
                break

    # ── PRIMARY INPUT — STEP 1 (outside form for immediate reactivity) ────────
    primary_key = phase.get("primary_input")
    if primary_key:
        primary_inp = next((i for i in phase["inputs"] if i["key"] == primary_key), None)
        if primary_inp:
            st.subheader("Step 1: Upload Phase 1 Output")
            _render_primary_input_section(phase_id, primary_inp, saved_inputs)
            st.markdown("---")
            st.subheader("Step 2: Review & Complete Details")
            st.caption("Fields marked \\* are required. Click **Generate Prompt** when ready.")
    else:
        st.subheader("📝 Enter Details")
        st.caption("Fields marked \\* are required. Click **Generate Prompt** when ready.")

    # Warn about required fields that autofill couldn't populate (skip hidden + primary input fields)
    needs_manual = [
        inp["label"]
        for inp in phase["inputs"]
        if inp.get("required") and not inp.get("hidden")
        and inp["key"] != (phase.get("primary_input") or "")
        and not saved_inputs.get(inp["key"], "").strip()
    ]
    if needs_manual:
        st.warning(
            "The following required fields need to be filled in manually "
            "(previous phase data not available):\n"
            + "".join(f"\n- **{label}**" for label in needs_manual)
        )

    collected = {}

    with st.form(key=f"form_phase_{phase_id}"):
        for inp in phase["inputs"]:
            # Hidden fields and the primary input (handled in Step 1) pass through silently
            if inp.get("hidden") or (primary_key and inp["key"] == primary_key):
                collected[inp["key"]] = saved_inputs.get(inp["key"], inp.get("default", ""))
                continue

            if inp.get("required"):
                label = inp["label"] + " \\*"
            elif "(optional)" in inp["label"].lower():
                label = inp["label"]
            else:
                label = inp["label"] + " *(Optional)*"
            common = dict(help=inp.get("help", ""), placeholder=inp.get("placeholder", ""))

            field_value = saved_inputs.get(inp["key"]) or inp.get("default", "")

            if inp["type"] == "file_or_text":
                st.markdown(f"**{label}**")
                if inp.get("help"):
                    st.caption(inp["help"])
                tab_type, tab_upload = st.tabs(["✏️ Type / Paste", "📎 Upload File"])
                with tab_type:
                    val = st.text_area(
                        "Content",
                        value=field_value,
                        height=200,
                        placeholder=inp.get("placeholder", ""),
                        label_visibility="collapsed",
                        key=f"text_{phase_id}_{inp['key']}",
                    )
                with tab_upload:
                    st.caption("Upload a saved `.md` or `.txt` file. Takes priority over typed content above.")
                    file_obj = st.file_uploader(
                        f"Upload {inp['label']}",
                        type=["md", "txt"],
                        key=f"__upload_{phase_id}_{inp['key']}",
                        label_visibility="collapsed",
                    )
                collected[inp["key"]] = val
                collected[f"__upload_{inp['key']}"] = file_obj

            elif inp["type"] == "textarea":
                val = st.text_area(label, value=field_value, height=100, **common)
                collected[inp["key"]] = val
            else:
                val = st.text_input(label, value=field_value, **common)
                collected[inp["key"]] = val

        generate_clicked = st.form_submit_button(
            "⚡ Generate Prompt", type="primary", use_container_width=True
        )

    if generate_clicked:
        # File upload takes priority over typed content; pop sentinels before saving to session state
        for inp in phase["inputs"]:
            if inp["type"] == "file_or_text":
                file_obj = collected.pop(f"__upload_{inp['key']}", None)
                if file_obj is not None:
                    try:
                        collected[inp["key"]] = file_obj.read().decode("utf-8")
                    except Exception as exc:
                        st.error(f"Could not read uploaded file for '{inp['label']}': {exc}")
                        collected[inp["key"]] = ""

        st.session_state.inputs[str(phase_id)] = collected

        missing = [
            inp["label"]
            for inp in phase["inputs"]
            if inp.get("required") and not inp.get("hidden")
            and not collected.get(inp["key"], "").strip()
        ]
        if missing:
            for label in missing:
                st.error(f"Required field is empty: **{label}**")
        else:
            st.session_state.prompts_shown[str(phase_id)] = True

    # ── PROMPT OUTPUT ──────────────────────────────────────────────
    if not st.session_state.prompts_shown.get(str(phase_id), False) and not done:
        return

    prompt = generate_prompt(phase_id)
    original_prompt = prompt

    cfg = st.session_state.llm_config
    has_key = bool(cfg.get("api_key", "").strip())
    provider = cfg.get("provider", "—")
    model = cfg.get("model", "—")

    st.markdown("---")
    st.subheader("📄 Generated Prompt")
    st.caption(
        "The prompt below contains only your descriptions — no raw data. "
        "It is safe to share with any LLM service."
    )

    with st.expander("View / Copy Prompt", expanded=True):
        st.code(prompt, language="markdown")

    st.download_button(
        "⬇️ Download Prompt (.txt)",
        data=prompt,
        file_name=f"phase{phase_id}_prompt.txt",
        mime="text/plain",
    )

    # ── IMPROVE PROMPT (OPTIONAL) ───────────────────────────────────
    st.markdown("---")
    st.subheader("✨ Improve Prompt with LLM (Optional)")
    st.caption(
        "Optionally ask an LLM to refine this prompt for clarity and specificity "
        "before using it in the step below."
    )

    pid = str(phase_id)
    improved_entry = None
    want_improve = st.checkbox(
        "Improve this prompt with an LLM before using it",
        value=st.session_state.use_improved_prompt.get(pid, False),
        key=f"improve_toggle_{phase_id}",
    )
    st.session_state.use_improved_prompt[pid] = want_improve

    if want_improve:
        default_improvement_prompt = build_prompt_improvement_request(prompt)

        with st.expander("View / Edit Improvement Request", expanded=True):
            st.caption("This is the request that will be sent to the LLM to improve your prompt. Edit it if needed.")
            improvement_prompt = st.text_area(
                "Improvement request prompt",
                value=default_improvement_prompt,
                height=250,
                key=f"improve_request_edit_{phase_id}",
                label_visibility="collapsed",
            )

        if st.session_state.mode == "API Mode" and has_key:
            if st.button("✨ Improve Prompt with LLM", key=f"improve_btn_{phase_id}"):
                with st.spinner(f"Calling {provider} to improve prompt..."):
                    try:
                        improved = call_llm(prompt=improvement_prompt, provider=provider, config=cfg)
                        st.session_state.improved_prompts[pid] = {
                            "content": improved.strip(),
                            "timestamp": datetime.now().isoformat(),
                            "source": "api",
                        }
                        save_session()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Prompt improvement failed: {exc}")
        else:
            if st.session_state.mode == "API Mode" and not has_key:
                st.caption("No API key configured — falling back to copy-paste for prompt improvement.")

            st.download_button(
                "⬇️ Download Improvement Request (.txt)",
                data=improvement_prompt,
                file_name=f"phase{phase_id}_prompt_improvement_request.txt",
                mime="text/plain",
                key=f"improve_dl_{phase_id}",
            )

            improved_paste = st.text_area(
                "Paste the improved prompt here",
                value=st.session_state.improved_prompts.get(pid, {}).get("content", ""),
                height=300,
                placeholder="Paste the improved prompt returned by your LLM service...",
                key=f"improve_paste_{phase_id}",
            )
            if st.button(
                "💾 Save Improved Prompt",
                key=f"improve_save_{phase_id}",
                disabled=not improved_paste.strip(),
            ):
                st.session_state.improved_prompts[pid] = {
                    "content": improved_paste.strip(),
                    "timestamp": datetime.now().isoformat(),
                    "source": "copy-paste",
                }
                save_session()
                st.rerun()

        if pid in st.session_state.improved_prompts:
            improved_entry = st.session_state.improved_prompts[pid]
            with st.expander("View Improved Prompt", expanded=True):
                st.code(improved_entry["content"], language="markdown")

            use_choice = st.radio(
                "Which version should be used below?",
                ["Original Prompt", "Improved Prompt"],
                index=1,
                key=f"prompt_choice_{phase_id}",
                horizontal=True,
            )
            if use_choice == "Improved Prompt":
                prompt = improved_entry["content"]

    if prompt != original_prompt:
        st.markdown("---")
        st.subheader("✅ Final Prompt to Use")
        st.caption(
            "This improved prompt will be used in the step below. Review it and make "
            "any final edits before copying it or sending it to the LLM."
        )

        final_prompt_key = f"final_prompt_edit_{phase_id}_{improved_entry['timestamp'] if improved_entry else ''}"
        prompt = st.text_area(
            "Final prompt to use",
            value=prompt,
            height=300,
            key=final_prompt_key,
            label_visibility="collapsed",
        )

        st.download_button(
            "⬇️ Download Final Prompt (.txt)",
            data=prompt,
            file_name=f"phase{phase_id}_prompt_final.txt",
            mime="text/plain",
            key=f"final_dl_{phase_id}",
        )

    st.markdown("---")

    # ── COPY-PASTE MODE ───────────────────────────────────────────
    if st.session_state.mode == "Copy-Paste Mode":
        st.subheader("💬 Submit LLM Response")
        st.markdown(
            "1. Copy the prompt above (use the **Final Prompt to Use** section if shown) "
            "and submit it to your preferred LLM service\n"
            "2. Paste the response below **or** upload the saved response file\n"
            "3. Click **Save Response** to store and continue"
        )

        existing_response = st.session_state.responses.get(str(phase_id), {}).get("content", "")

        tab_paste, tab_upload = st.tabs(["✏️ Paste Response", "📎 Upload File"])

        with tab_paste:
            response_text = st.text_area(
                "LLM Response",
                value=existing_response,
                height=400,
                placeholder="Paste the LLM response here...",
                label_visibility="collapsed",
                key=f"resp_paste_{phase_id}",
            )

        with tab_upload:
            st.caption("Upload a saved `.md` or `.txt` response file. Takes priority over pasted text.")
            uploaded_resp = st.file_uploader(
                "Upload response file",
                type=["md", "txt"],
                key=f"resp_upload_{phase_id}",
                label_visibility="collapsed",
            )

        # Resolve: upload takes priority over pasted text
        uploaded_content = ""
        if uploaded_resp is not None:
            try:
                uploaded_content = uploaded_resp.read().decode("utf-8")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
        final_response = uploaded_content if uploaded_content.strip() else response_text

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save Response", use_container_width=True, disabled=not final_response.strip()):
                st.session_state.responses[str(phase_id)] = {
                    "content": final_response,
                    "timestamp": datetime.now().isoformat(),
                    "source": "file-upload" if uploaded_content.strip() else "copy-paste",
                }
                filepath = save_output(final_response, phase["output_filename"], phase_id)
                save_session()
                st.success(f"Saved to `{filepath}`")
                st.rerun()

        with col2:
            if done:
                st.download_button(
                    "⬇️ Download Response",
                    data=st.session_state.responses[str(phase_id)]["content"],
                    file_name=phase["output_filename"],
                    mime="text/markdown",
                    use_container_width=True,
                )

        with col3:
            _render_next_button(phase_id, key_suffix="cp")

    # ── API MODE ──────────────────────────────────────────────────
    elif st.session_state.mode == "API Mode":
        st.subheader("🚀 Send to LLM API")

        col_info, col_send = st.columns([3, 1])
        with col_info:
            st.caption(
                f"Provider: **{provider}** | Model: **{model}** | "
                f"API Key: {'✅ Set' if has_key else '❌ Not set (configure in sidebar)'}"
            )
        with col_send:
            send_clicked = st.button(
                "🚀 Send to LLM",
                type="primary",
                use_container_width=True,
                disabled=not has_key,
            )

        if not has_key:
            st.warning("Enter your API key in the sidebar to use API Mode.")

        with st.expander("📎 Upload a saved response instead"):
            st.caption("Use this to load a previously saved `.md` or `.txt` response for this phase, without calling the API.")
            api_uploaded = st.file_uploader(
                "Upload response file",
                type=["md", "txt"],
                key=f"api_resp_upload_{phase_id}",
                label_visibility="collapsed",
            )
            if api_uploaded is not None:
                try:
                    api_content = api_uploaded.read().decode("utf-8")
                    st.success(f"Loaded **{api_uploaded.name}**")
                    if st.button("💾 Save Uploaded Response", key=f"save_api_upload_{phase_id}", type="primary"):
                        st.session_state.responses[str(phase_id)] = {
                            "content": api_content,
                            "timestamp": datetime.now().isoformat(),
                            "source": "file-upload",
                        }
                        filepath = save_output(api_content, phase["output_filename"], phase_id)
                        save_session()
                        st.success(f"Saved to `{filepath}`")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")

        if send_clicked and has_key:
            with st.spinner(f"Calling {provider}..."):
                try:
                    response = call_llm(prompt=prompt, provider=provider, config=cfg)
                    st.session_state.responses[str(phase_id)] = {
                        "content": response,
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider,
                        "model": model,
                    }
                    filepath = save_output(response, phase["output_filename"], phase_id)
                    save_session()
                    st.success(f"Response received and saved to `{filepath}`")
                    st.rerun()
                except Exception as exc:
                    st.error(f"API call failed: {exc}")
                    st.caption("Check your API key, model name, and network connection.")

        if done:
            resp = st.session_state.responses[str(phase_id)]
            st.markdown("---")
            st.subheader("📤 LLM Response")
            st.caption(
                f"Source: {resp.get('provider', 'manual')} | "
                f"Model: {resp.get('model', '—')} | "
                f"Saved: {resp.get('timestamp', '—')}"
            )
            with st.expander("View Response", expanded=True):
                st.markdown(resp["content"])

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Response",
                    data=resp["content"],
                    file_name=phase["output_filename"],
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                _render_next_button(phase_id, key_suffix="api")


def _render_next_button(phase_id: int, key_suffix: str = ""):
    next_phase = phase_id + 1
    if next_phase in PHASES:
        if st.button(
            f"Next: {PHASES[next_phase]['icon']} Phase {next_phase} ▶",
            use_container_width=True,
            type="primary",
            key=f"next_{phase_id}_{key_suffix}",
        ):
            st.session_state.current_phase = next_phase
            st.rerun()
    else:
        if st.button(
            "🎉 Pipeline Complete!",
            use_container_width=True,
            type="primary",
            key=f"done_{phase_id}_{key_suffix}",
        ):
            st.balloons()
            st.success("All phases complete! Your ML pipeline is ready for iteration.")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="ML Pipeline Guide",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stButton > button { border-radius: 6px; }
        .stTextArea > div > textarea { font-size: 0.85em; }
        [data-testid="stExpander"] summary p { font-size: 1.5rem !important; font-weight: 600 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session()
    render_sidebar()

    phase_id = st.session_state.current_phase
    if phase_id == 0:
        render_overview()
    elif phase_id in PHASES:
        render_phase(phase_id)
    else:
        st.error(f"Unknown phase: {phase_id}")
        st.session_state.current_phase = 0
        st.rerun()


if __name__ == "__main__":
    main()
