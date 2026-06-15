# ML Pipeline Guide — How to Run

## What This Tool Does

A locally-running web app that guides data scientists through an end-to-end
LLM-assisted ML development pipeline. Your sensitive data never leaves your machine —
only text descriptions you write are sent to LLMs.

---

## Prerequisites

- **Python 3.9 or later** — check with `python --version`
- **pip** — check with `pip --version`
- An internet connection (only needed for API Mode or to install dependencies)

---

## Step 1 — Create a Virtual Environment (Recommended)

A virtual environment keeps this project's dependencies isolated from your system Python.

Open a terminal (Command Prompt or PowerShell) and navigate to this folder:

```
cd "d:\Synapxe Project\2026 LLM Code with Sensitive Data\ml_pipeline_guide"
```

Create the virtual environment:

```
python -m venv sii
```

Activate it:

- **PowerShell:** `.\sii\Scripts\Activate.ps1`
- **Command Prompt:** `sii\Scripts\activate.bat`

You should see `(venv)` at the start of your prompt once it is active.

> **Note (PowerShell only):** If you get an error about execution policy, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry activation.

---

## Step 2 — Install Dependencies

With the virtual environment active, install the required packages:

```
pip install -r requirements.txt
```

This installs:
- `streamlit` — the web UI framework
- `openai` — for OpenAI and Azure OpenAI API calls (API Mode only)
- `anthropic` — for Anthropic Claude API calls (API Mode only)

---

## Step 3 — Launch the App

From the same folder, run:

```
streamlit run app.py
```

Or double-click **`run.bat`**.

Streamlit will print something like:

```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser should open automatically. If it does not, paste the Local URL
into any browser.

---

## Step 4 — Choose Your Interaction Mode

In the **left sidebar**, select one of two modes before starting:

### Copy-Paste Mode (recommended for sensitive data)
- The tool generates a filled-in prompt for each pipeline phase.
- You **copy** the prompt and paste it into any LLM chatbot:
  - Claude — https://claude.ai
  - ChatGPT — https://chat.openai.com
  - Gemini — https://gemini.google.com
  - Any other LLM interface
- You **paste the response back** into the tool.
- Nothing is sent over the internet automatically.

### API Mode
- The tool sends the prompt directly to an LLM API.
- Configure in the sidebar:
  1. Select provider: **OpenAI**, **Anthropic (Claude)**, or **Azure OpenAI**
  2. Enter your API key (stored in your browser session only — never written to disk)
  3. Select the model
- Click **Send to LLM** on each phase page.

---

## Step 5 — Work Through the Pipeline

The pipeline has 7 phases. Navigate using the sidebar buttons.

| Phase | What You Provide | What the LLM Returns |
|-------|-----------------|----------------------|
| 🎯 Phase 1 — Problem Definition | Objective, target variable, data sources, constraints, metrics | Refined ML problem statement (markdown) |
| 📚 Phase 2 — Literature Review | Phase 1 output, domain, task type | Model recommendations + SOTA literature review |
| 🔍 Phase 3 — EDA Planning | Problem statement, dataset description, folder paths | Complete EDA Jupyter Notebook (.ipynb) |
| 💡 Phase 3b — EDA Insights | Problem statement, EDA analysis results | Structured insights for modelling |
| ⚙️ Phase 4 — Preprocessing | Problem statement, EDA insights, data description | Preprocessing pipeline notebook |
| 🧪 Phase 5 — Experiment Design | Literature review, EDA insights, evaluation metrics | Experiment strategy document |
| 🚀 Phase 6 — Training Pipeline | Problem statement, experiment plan, column names | Full modelling notebook with experiment tracking |

**Typical flow for each phase:**
1. Fill in the input form fields
2. Click **Generate Prompt**
3. Copy-paste the prompt to your LLM (or click Send to LLM in API Mode)
4. Paste the response back (Copy-Paste Mode) or view the API response
5. Click **Save Response**, then **Next Phase**

---

## Saving and Resuming Your Work

### Auto-save
After each **Save Response** action, progress is automatically written to
`pipeline_state.json` in the same folder.

### Manual save
Click **💾 Save Progress** in the sidebar at any time.

### Export your session
Click **⬇️ Export Session JSON** to download a portable session file you can
share or back up.

### Resume a session
Click **📂 Load Session JSON** in the sidebar and select a previously exported
`.json` file.

---

## Where Outputs Are Saved

Each saved LLM response is written to:

```
ml_pipeline_guide/
└── outputs/
    ├── phase1/  phase1_problem_definition.md
    ├── phase2/  phase2_literature_review.md
    ├── phase3/  phase3_eda_notebook_request.md
    ├── phase4/  phase3b_eda_insights.md
    ├── phase5/  phase4_preprocessing_plan.md
    ├── phase6/  phase5_experiment_plan.md
    └── phase7/  phase6_training_pipeline_request.md
```

These markdown files can be copied into subsequent phase inputs to carry context forward.

---

## Stopping the App

Press **Ctrl + C** in the terminal window where Streamlit is running.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit: command not found` | Run `pip install streamlit` then retry |
| Browser does not open | Manually open `http://localhost:8501` |
| API call fails with auth error | Check that the API key is correct and has available quota |
| API call fails with model not found | Verify the model name matches your account's access |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| `pip install` installs to wrong environment | Activate your virtual environment first |

---

## API Key Reference

| Provider | Where to get the key |
|----------|---------------------|
| OpenAI | https://platform.openai.com/api-keys |
| Anthropic (Claude) | https://console.anthropic.com/settings/keys |
| Azure OpenAI | Azure Portal → your OpenAI resource → Keys and Endpoint |

> API keys are held in your browser session only. They are never written to
> `pipeline_state.json` or any output file.
