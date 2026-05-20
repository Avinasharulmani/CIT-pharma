# Pharma Intelligence Engine

This project merges the existing key-message mapping work with audio/video extraction logic and provides a React UI served by FastAPI.

## Entry points

- **React UI**: served by FastAPI at `/`
- **FastAPI API**: JSON API at `/api/analyze`
- **The code should be base model**: all outputs are returned using reusable Pydantic BaseModel classes in `pharma_intelligence/models.py`.

Enhance work to build a reusable library which can take in different type of Pharma commercial materials **PPT, PDF, Video, Image, Audio** to provide extracted intelligence from those documents. This will be later used for **MLR Review, Key Message Mapping** and similar pharma commercial review use cases. The output provides text-based intelligence for all uploaded documents. The program is written as a modular Python package. The document may be in multiple languages, and the processor can detect language and perform optional translation before analytics and mapping.

## Workflow rules followed

The code follows the workflow diagram:

1. React UI receives material upload, target language, optional page/slide, and mapping flag.
2. Orchestration detects file type.
3. File-specific processor runs:
   - PDF: page-level text extraction
   - PPT: slide-level text extraction
   - Image: OCR + image description
   - Audio: Whisper base-model transcription
   - Video: extract audio, transcribe audio, sample frames, OCR frame text, combine video intelligence
4. Shared intelligence services clean and normalize text.
5. Language detection and translation run when required.
6. Analytics layer generates summary and KeyBERT keywords.
7. Key Message Mapping Engine loads Excel/CSV key messages, creates embeddings, matches extracted text with key messages.
8. Response model returns structured output.

## UI output columns

The UI shows the mapping table with:

- Key Message ID
- Brand/Product
- Key Message
- Key Words, generated using KeyBERT with fallback keyword extraction
- Source
- Confidence
- Description of Image/Video, populated only for images and videos

## Folder structure

```text
pharma_intelligence_fastapi/
├── app.py
├── requirements.txt
├── README.md
├── WORKFLOW_RULES.md
├── sample_key_messages.csv
├── pharma_intelligence/
│   ├── config.py
│   ├── models.py
│   ├── utils.py
│   ├── language.py
│   ├── keywords.py
│   ├── key_messages.py
│   ├── extractors/
│   │   ├── pdf_processor.py
│   │   ├── ppt_processor.py
│   │   ├── image_processor.py
│   │   ├── audio_processor.py
│   │   └── video_processor.py
│   └── services/
│       ├── intelligence_service.py
│       └── mapping_service.py
├── uploads/
└── outputs/
```

## Setup

```bash
cd pharma_intelligence_fastapi
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install external tools:

```bash
winget install Gyan.FFmpeg
```

For OCR, install Tesseract OCR and add it to PATH.

## Run React UI/API

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Key message file

You can upload your Excel/CSV in the UI, or place your file in the project root with this name:

```text
Key_Messages_Final_NEW.xlsx
```

Expected columns can be flexible, but these are recommended:

```text
Key Message ID, Brand/Product, Key Message
```

## Notes

- Audio/video transcription uses Whisper `base` by default in `config.py`.
- Video processing samples frames every 10 seconds for faster results.
- If KeyBERT or sentence-transformers is not available, fallback keyword/vector logic is used.
- For image description, BLIP base captioning is attempted if `transformers` is installed; otherwise the system uses a safe base description from image metadata and OCR text.
