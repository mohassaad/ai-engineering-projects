Week 2 - LLM Extraction Project
=================================
Mohamed Assaad

WHAT THIS PROJECT DOES
-----------------------
Two tasks using the LLaMA 3.3-70B model via the Groq API:

Task 1 - Text Classifier (classifier.py / classifier_task.ipynb)
  Classifies any English sentence into exactly one of five categories:
  Sports, Politics, Technology, Health, or Entertainment.

Task 2 - Information Extraction (information_extraction_task.py / information_extraction_task.ipynb)
  Extracts five structured fields from any English sentence:
  Person, Organization, Location, Date, and Event.
  Writes N/A for any field not found in the text.

Both tasks were tested on 5 sample sentences each and returned correct results.

FOLDER STRUCTURE
-----------------
Week2_llm_extraction_proj/
├── classifier.py                           → Task 1 Python script
├── classifier_task.ipynb                   → Task 1 Jupyter notebook
├── Classifier Task Summary.txt             → Task 1 summary
├── information_extraction_task.py          → Task 2 Python script
├── information_extraction_task.ipynb       → Task 2 Jupyter notebook
└── Information Extraction Task Summary.txt → Task 2 summary

REQUIREMENT
------------
Internet connection is required — the model runs on Groq's servers, not locally.

HOW TO RUN — VS CODE WITH VENV
--------------------------------
Run these commands one by one in the terminal:

    cd ..
    cd Week2_llm_extraction_proj
    dir
    python -m venv venv
    venv\Scripts\activate
    pip install groq
    python classifier.py

To run Task 2 after Task 1:
    python information_extraction_task.py

Results will appear in the terminal.

HOW TO RUN — JUPYTER NOTEBOOK WITH ANACONDA
---------------------------------------------
Step 1 - Open Command Prompt and run:
    jupyter notebook

Step 2 - Navigate to the Week2_llm_extraction_proj folder

Step 3 - For Task 1: open classifier_task.ipynb
         For Task 2: open information_extraction_task.ipynb

Step 4 - Run each cell with Shift + Enter

MODEL USED
-----------
LLaMA 3.3-70B via Groq API (free tier)
API key is included in the code and lasts 90 days from creation.

LIMITATIONS
------------
- Requires internet connection to call Groq API
- Classifier: sentences that belong to multiple categories may be
  misclassified since the model must pick only one label
- Extractor: informal sentences with no clear named entities may
  return N/A for most fields