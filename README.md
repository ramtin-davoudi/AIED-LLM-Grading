# AIED Project: LLM-based Automatic Short Answer Grading

## Overview

This project evaluates the performance of large language models (LLMs) for automatic short-answer grading using the SciEntsBank dataset (SemEval-2013 Task 7). We compare GPT-4o and GPT-5.2 under different prompting strategies (zero-shot, 5-shot, and 15-shot) with a classical NLP baseline based on a bigram language model.

The project includes:
- Data processing
- LLM-based evaluation using OpenAI Batch API
- Baseline implementation
- Quantitative evaluation (accuracy, macro F1, weighted F1)
- Qualitative error analysis and visualization

------------------------------------------------------------------------------------

## Project Structure

├── analysis_results/ # Final evaluation outputs and summaries for LLMs
├── bigram_baseline_results/ # Results from bigram baseline model
├── scientsbank_batch_run/ # Raw outputs from OpenAI Batch API runs
│
├── zero_shot.py # GPT zero-shot evaluation
├── five_shot.py # GPT 5-shot evaluation
├── fifteen_shot.py # GPT 15-shot evaluation
│
├── bigram_baseline.py # Classical NLP baseline implementation
├── analysis.py # Evaluation + metrics computation
│
├── figures.py # Plot generation (charts, confusion matrices)
│
├── cm_gpt4o_5shot.eps # Confusion matrix (GPT-4o)
├── cm_gpt52_5shot.eps # Confusion matrix (GPT-5.2)
├── model_comparison.eps # Model comparison chart
├── per_class_comparison.eps # Per-class performance chart
│
├── api_key.txt # OpenAI API key




How to Run
1. Set API Key

Insert your OpenAI API key in OPENAI_API_KEY for zero_shot.py, five_shot.py, and fifteen_shot.py.

2. Run LLM Experiments

Zero-shot:

python zero_shot.py

5-shot:

python five_shot.py

15-shot:

python fifteen_shot.py

These scripts will:

- Load the dataset
- Generate prompts
- Send requests via OpenAI Batch API
- Save outputs in scientsbank_batch_run/

3. Run Bigram Baseline
python bigram_baseline.py

This script:

- Trains a bigram model with smoothing
- Predicts labels
- Saves results in bigram_baseline_results/

4. Run Evaluation
python analysis.py

This computes:

- Accuracy
- Precision / Recall
- Macro F1
- Weighted F1
- Per-class metrics
- Confusion matrices

- Results are saved in analysis_results/.

5. Generate Figures
python figures.py

This generates:

- Model comparison plots
- Per-class performance plots
- Confusion matrices

All figures are saved in .eps format for LaTeX use.

Notes on Reproducibility
This project relies on the OpenAI API for LLM inference.
Running the full pipeline may require API access and may incur cost.
Precomputed outputs are already included in:
scientsbank_batch_run/
analysis_results/

These can be used to reproduce the results without rerunning the API calls.