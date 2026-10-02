# StudySnap AI

StudySnap AI is a beginner-friendly Streamlit application that turns a screenshot of lecture notes into a simple revision sheet. It reads text with local OCR, cleans the result, finds likely keywords, and creates practice questions without paid APIs or API keys.

## Problem statement

Students often keep important class notes in screenshots, where the text is harder to search, summarize, and revise. StudySnap AI converts a notes image into readable text and organizes it into a useful starting point for revision.

## Objectives

- Read text from JPG, JPEG, and PNG notes screenshots.
- Show both the OCR output and its cleaned version.
- Identify a possible topic and important terms with a simple NLP method.
- Produce structured notes and practice questions.
- Keep the workflow understandable and use free, local tools.

## Features

- Image upload and preview.
- OCR with Pillow and pytesseract.
- Text normalization that removes extra spaces and empty lines.
- OCR cleanup for stray bars, isolated punctuation, and numbering-only noise while preserving numbered list items.
- Keyword ranking with scikit-learn TF-IDF, plus filters for generic instruction words.
- Topic detection that prefers recognizable technical subjects over arbitrary OCR fragments.
- A concise extractive summary, ranked important points, and a quick revision checklist.
- Rule-based Theory, Programming, Practical, or Mixed study-type detection.
- Exam Focus suggestions based on the extracted terms and content.
- Exam Mode with five topic-related questions, three short-answer questions, and three content-based MCQs.
- Save the notes to `output/studysnap_notes.txt` or download a copy.
- Helpful messages for invalid images, unreadable OCR results, and missing Tesseract.

## Technologies

- **Python** for application logic.
- **Streamlit** for the interactive web application.
- **Pillow** for opening and validating images.
- **pytesseract** for calling the OCR engine.
- **Tesseract OCR** for local text recognition.
- **scikit-learn** for the TF-IDF keyword ranking.
- **pandas** and **NumPy** are included in the project requirements for learning and future data work.

## How OCR works

1. The user uploads a JPG, JPEG, or PNG image.
2. Pillow opens the image and checks that its contents are readable.
3. pytesseract passes the image to the installed Tesseract OCR program.
4. Tesseract recognizes characters and returns text.
5. StudySnap AI normalizes Unicode, reduces repeated whitespace, and removes blank lines.

The Python package `pytesseract` is a connector; it does not install the Tesseract OCR program itself. Install Tesseract separately and ensure it is available on your system PATH. On Windows, if Tesseract is installed in a non-standard location, configure its executable path in your local setup before running OCR.

## NLP approach

StudySnap AI uses a small, explainable NLP pipeline:

- **Cleaning:** Unicode normalization and whitespace cleanup make OCR output easier to read.
- **Keywords:** `TfidfVectorizer` ranks useful unigrams and bigrams while ignoring common English stop words and generic instruction words. A short technical vocabulary helps prioritize recognizable terms. For one document, this is a lightweight heuristic; it does not understand the subject like a language model.
- **Topic:** recognizable technical subject terms are preferred; if none are found, the app uses useful keywords rather than an arbitrary opening OCR fragment.
- **Summary and important points:** complete, distinct sentences are ranked using keyword matches and sentence length. HTML notes receive a concise subject summary based on the detected elements and practice tasks rather than isolated OCR fragments.
- **Study type and exam focus:** simple word cues classify the material and surface extracted technical terms for revision.

## Exam Mode

All practice questions are generated locally from the extracted text and keywords:

- Five questions are generated. Programming notes favor practical coding prompts, and recognized HTML exercises (such as webpages, tables, forms, and ordered lists) are used directly as the question basis.
- Three short-answer prompts ask about extracted concepts.
- Three MCQs test concepts found in the notes; HTML questions use the relevant element or control and provide the correct answer for checking.
- Exam Focus prioritizes technical concepts such as HTML structure, heading tags, tables, lists, forms, and input controls over incidental field names.

These are rule-based practice prompts, not guaranteed exam questions. Check the wording and answer choices against the original notes.

## Project structure

```text
StudySnap-AI/
├── data/                 # Optional datasets or sample inputs
├── notebooks/            # Optional experiments and learning notebooks
├── src/
│   └── app.py            # Main Streamlit application
├── uploads/              # Local uploads (ignored by Git)
├── output/               # Saved notes (ignored by Git)
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Install Python 3.10 or newer.
2. Install the Tesseract OCR program for your operating system.
3. Open a terminal in the project folder.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install the Python packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

## How to run

From the project folder, run:

```powershell
streamlit run src/app.py
```

Streamlit prints a local URL in the terminal; open it in your browser. Keep the terminal window open while using the app.

## Example workflow

1. Take a clear screenshot of a lecture slide or handwritten/typed notes.
2. Open StudySnap AI in your browser and upload the image in the sidebar.
3. Check the image preview and select **Analyze screenshot**.
4. Review the OCR text and cleaned text; OCR may need correction.
5. Read the topic, summary, keywords, important points, and revision checklist.
6. Use the questions and MCQs to test your understanding.
7. Select **Save notes to output/** or **Download notes** to keep the revision sheet.

## Limitations

- OCR accuracy depends on image sharpness, contrast, language, orientation, and text size.
- Tesseract must be installed separately; installing the Python requirements alone is not sufficient.
- The keyword and topic methods are simple heuristics, not advanced semantic understanding.
- The summary selects opening sentences and may miss important details later in the notes.
- Generated questions and MCQs can be repetitive or inaccurate. Check them against your original material.
- This project does not store a database of students or uploaded files.

## Future improvements

- Add image preprocessing for rotation, contrast, and noise reduction.
- Support multiple OCR languages and handwritten notes more reliably.
- Improve topic detection and summaries with optional open-source models.
- Add editable notes, PDF export, and question difficulty levels.
- Add tests and sample images for the OCR and text-processing functions.
