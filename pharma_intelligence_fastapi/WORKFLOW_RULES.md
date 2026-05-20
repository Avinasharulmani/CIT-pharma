# Workflow Rules Implemented

The application follows the provided Pharma Intelligence Engine flow as implementation rules.

## Flow

```text
FastAPI UI Layer
  -> Upload Pharma Commercial Material
  -> Accept target language, optional page/slide, and mapping flag
  -> Orchestration Engine detects file type
  -> PDF/PPT/Image/Audio/Video processor
  -> Shared Intelligence Services
  -> Translation Service when needed
  -> Analytics & Mapping Layer
  -> Key Message Mapping Engine when mapping is enabled
  -> Base Response Model
  -> FastAPI Output
```

## Processor rules

- PDF prepares page-level text intelligence.
- PPT prepares slide-level text intelligence.
- Image prepares OCR text and image description.
- Audio prepares document-level text intelligence through transcription.
- Video extracts audio from video, transcribes the audio, samples frames, runs OCR on sampled frames, and combines transcript plus frame text.

## Mapping rules

- Key messages are loaded from Excel/CSV.
- KeyBERT generates keywords.
- Extracted content is mapped against brand/product and key message text.
- Results are returned as structured BaseModel output.

## UI rules

The FastAPI UI must show:

- Key Message ID
- Brand/Product
- Key Message
- Key Words
- Description of Image/Video only for images and videos
