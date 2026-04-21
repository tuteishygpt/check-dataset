#  TTS Dataset Validator

## Vertex AI setup

This project now uses Gemini through Vertex AI with Application Default Credentials instead of `GOOGLE_API_KEY`.

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global
export VERTEX_BATCH_GCS_URI=gs://your-bucket/tts-validator
```

`Flex PayGo` on Vertex AI uses HTTP headers, works only on the `global`
endpoint, and is currently limited to Gemini 3 preview models. This project
uses the documented header-based flow instead of `service_tier=flex`.

`Batch Mode` uses the real Vertex Batch API with JSONL requests in Cloud
Storage. The app uploads audio as WAV files to `VERTEX_BATCH_GCS_URI`, creates
all required batch jobs immediately in groups of 100 files, waits for
completion, and merges the output back into the normal analysis results.

Інструмент для аналізу аўдыядатасетаў з мэтай выяўлення несупадзенняў паміж метаданымі (тэкстам) і рэальным гукам. Выкарыстоўвае Gemini праз Vertex AI для транскрыбацыі аўдыя.

## ✨ Магчымасці

- 🎤 **Транскрыбацыя аўдыя** з дапамогай Gemini API
- 📊 **Параўнанне вынікаў** розных мадэлей
- 🧠 **Разумны аналіз** з паслядоўным выкарыстаннем некалькіх мадэлей
- 📦 **Vertex AI сінхронны рэжым** з inline-аўдыя без upload/batch API
- 📥 **Імпарт/экспарт** вынікаў у CSV
- 🤗 **Стварэнне датасэтаў** на Hugging Face з правераных запісаў
- ✅ **Ручная верыфікацыя** праблемных запісаў

## 🚀 Запуск

```bash
# Усталяванне залежнасцей
pip install -r requirements.txt

# Запуск праграмы
python app.py
```

Праграма адкрыецца ў браўзеры на `http://localhost:7860`

## 📁 Структура праекта

```
check dataset/
├── app.py                    # Entry point (27 радкоў)
│
├── core/                     # Асноўная логіка
│   ├── __init__.py
│   ├── state.py              # Глабальны стан (global_results, dataset_cache)
│   ├── cache.py              # Кэшаванне датасетаў
│   └── comparison.py         # Параўнанне вынікаў мадэлей
│
├── analysis/                 # Аналітычныя модулі
│   ├── __init__.py
│   ├── standard.py           # Стандартны аналіз праз Vertex AI
│   ├── smart.py              # Разумны аналіз (мульці-мадэльны)
│   └── import_export.py      # Імпарт/экспарт CSV, стварэнне датасэтаў
│
├── ui/                       # Карыстальніцкі інтэрфейс
│   ├── __init__.py
│   ├── audio.py              # Канвертацыя аўдыя ў base64
│   ├── dashboard.py          # Генерацыя HTML дашборду
│   ├── styles.py             # CSS стылі і JavaScript
│   └── gradio_app.py         # Gradio UI кампаненты
│
├── gemini_api.py             # Інтэграцыя з Gemini API
├── utils.py                  # Утыліты (загрузка датасетаў, нармалізацыя)
└── requirements.txt          # Залежнасці Python
```

## 📋 Модулі

### `core/` - Асноўная логіка

| Файл | Апісанне |
|------|----------|
| `state.py` | Глабальны стан праграмы: `global_results` (спіс вынікаў), `dataset_cache` (кэш датасетаў) |
| `cache.py` | Функцыі кэшавання: `get_cached_dataset()`, `cache_dataset()` |
| `comparison.py` | Параўнанне мадэлей: `select_best_model_result()`, `find_best_model_pair()`, `get_all_model_comparison()` |

### `analysis/` - Аналітычныя модулі

| Файл | Апісанне |
|------|----------|
| `standard.py` | `run_analysis()` - стандартны аналіз праз Vertex AI і пераправерка праблемных файлаў |
| `smart.py` | `run_smart_analysis()` - разумны аналіз з паслядоўным выкарыстаннем 4 мадэлей (Flash-Lite → Flash → Gemini-3-Flash) |
| `import_export.py` | Імпарт CSV, экспарт вынікаў, верыфікацыя запісаў, стварэнне датасэтаў на HuggingFace |

### `ui/` - Карыстальніцкі інтэрфейс

| Файл | Апісанне |
|------|----------|
| `audio.py` | `array_to_b64_audio()` - канвертацыя numpy array у base64 HTML audio tag |
| `dashboard.py` | `generate_dashboard_outputs()` - генерацыя HTML статыстыкі і спісу праблемных файлаў |
| `styles.py` | CSS стылі і JavaScript код для Gradio UI |
| `gradio_app.py` | `create_interface()` - стварэнне Gradio інтэрфейсу з усімі кампанентамі |

### Каранёвыя файлы

| Файл | Апісанне |
|------|----------|
| `app.py` | Entry point - загружае .env і запускае Gradio |
| `gemini_api.py` | `GeminiIntegrator` - клас для Vertex AI Gemini з inline WAV-аўдыя |
| `utils.py` | Загрузка датасетаў з HuggingFace, нармалізацыя тэксту, вылічэнне падабенства |

## ⚙️ Канфігурацыя

Стварыце файл `.env` у каранёвай тэчцы:

```env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_APPLICATION_CREDENTIALS=path-to-service-account.json
VERTEX_BATCH_GCS_URI=gs://your-bucket/tts-validator
# Flex PayGo on Vertex AI requires `global` and supported Gemini 3 preview models.
```

Аўтэнтыфікацыя ідзе праз Vertex AI ADC, API key больш не выкарыстоўваецца.

## 📊 Працоўны працэс

1. **Наладзьце Vertex AI ADC** і імя датасету HuggingFace
2. **Выберыце мадэль** і ўсталюйце параметры (ліміт файлаў, парог)
3. **Запусціце аналіз** (звычайны або разумны)
4. **Праглядзіце вынікі** - праблемныя файлы будуць пазначаны
5. **Верыфікуйце ўручную** або прыміце прапанаваныя выпраўленні
6. **Экспартуйце вынікі** ў CSV або стварыце новы датасэт на HuggingFace

## 🔧 Залежнасці

- `gradio` - вэб-інтэрфейс
- `google-genai` - Gemini API
- `datasets` - HuggingFace datasets
- `soundfile`, `librosa` - апрацоўка аўдыя
- `rapidfuzz` - вылічэнне падабенства тэксту
- `pandas` - праца з данымі

## 📝 Ліцэнзія

MIT License
