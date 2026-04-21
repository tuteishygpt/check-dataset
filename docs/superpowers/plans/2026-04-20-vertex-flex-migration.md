# Vertex Flex Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Gemini Developer API usage with Vertex AI via ADC and replace Batch Mode with Flex inference across the app.

**Architecture:** Keep the existing analysis flows and UI wiring, but swap the Google integration layer to a Vertex-first client that uses inline audio parts and `service_tier="flex"` when enabled. Remove file-upload batch jobs, registry state, and API-key-based Gemini setup while preserving the current sync/smart analysis behavior and Gradio UX.

**Tech Stack:** Python, Gradio, google-genai, Vertex AI ADC, unittest, soundfile

---

## Chunk 1: Vertex Integration Layer

### Task 1: Add failing tests for Vertex/Flex config helpers

**Files:**
- Create: `D:\googlePRJ\check dataset\tests\test_gemini_api.py`
- Modify: `D:\googlePRJ\check dataset\gemini_api.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_generation_config_enables_flex_timeout():
    config = build_generation_config(temperature=0.3, thinking_budget=0, flex_mode=True)
    assert config["service_tier"] == "flex"
    assert config["http_options"].timeout >= 600000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemini_api -v`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_generation_config(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemini_api -v`
Expected: PASS for the new helper test.

### Task 2: Replace API-key Gemini client with Vertex ADC client

**Files:**
- Modify: `D:\googlePRJ\check dataset\gemini_api.py`
- Test: `D:\googlePRJ\check dataset\tests\test_gemini_api.py`

- [ ] **Step 1: Write failing tests for Vertex env validation and client init**
- [ ] **Step 2: Run targeted unittest command and confirm failure**
- [ ] **Step 3: Implement Vertex-only client initialization using env or explicit project/location**
- [ ] **Step 4: Run targeted unittest command and confirm pass**

### Task 3: Replace file-upload batch logic with inline audio transcription helpers

**Files:**
- Modify: `D:\googlePRJ\check dataset\gemini_api.py`
- Test: `D:\googlePRJ\check dataset\tests\test_gemini_api.py`

- [ ] **Step 1: Write failing tests for audio re-encoding and task transcription path**
- [ ] **Step 2: Run targeted unittest command and confirm failure**
- [ ] **Step 3: Implement inline audio parts and remove file-registry/upload requests code**
- [ ] **Step 4: Run targeted unittest command and confirm pass**

## Chunk 2: Analysis Flow Migration

### Task 4: Rename Batch Mode to Flex inference in standard analysis

**Files:**
- Modify: `D:\googlePRJ\check dataset\analysis\standard.py`
- Test: `D:\googlePRJ\check dataset\tests\test_analysis_config.py`

- [ ] **Step 1: Write failing test for Flex-mode config path**
- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m unittest tests.test_analysis_config -v` and confirm failure**
- [ ] **Step 3: Update analysis flow to use `flex_mode` and Vertex config builder**
- [ ] **Step 4: Run the same unittest command and confirm pass**

### Task 5: Update smart analysis to use Vertex config builder

**Files:**
- Modify: `D:\googlePRJ\check dataset\analysis\smart.py`
- Test: `D:\googlePRJ\check dataset\tests\test_analysis_config.py`

- [ ] **Step 1: Write failing test for smart-analysis config generation**
- [ ] **Step 2: Run targeted unittest command and confirm failure**
- [ ] **Step 3: Remove direct `google.genai` config creation and reuse shared helper**
- [ ] **Step 4: Run targeted unittest command and confirm pass**

## Chunk 3: UI and Docs

### Task 6: Replace API key and batch UI with Vertex/Flex messaging

**Files:**
- Modify: `D:\googlePRJ\check dataset\ui\gradio_app.py`
- Modify: `D:\googlePRJ\check dataset\ui\styles.py`

- [ ] **Step 1: Write failing test for UI defaults or helper behavior where practical**
- [ ] **Step 2: Run targeted unittest command if added**
- [ ] **Step 3: Remove Gemini API key dependency from UI, rename Batch Mode to Flex inference, update localStorage key names**
- [ ] **Step 4: Run relevant unittest command or lightweight import check**

### Task 7: Update README and app descriptions

**Files:**
- Modify: `D:\googlePRJ\check dataset\README.md`
- Modify: `D:\googlePRJ\check dataset\app.py`
- Modify: `D:\googlePRJ\check dataset\requirements.txt`

- [ ] **Step 1: Update docs from Gemini API key / Batch API wording to Vertex AI / Flex inference**
- [ ] **Step 2: Remove stale dependency references and obsolete registry/batch descriptions**
- [ ] **Step 3: Run a lightweight import smoke test**

## Chunk 4: Verification

### Task 8: Run regression verification

**Files:**
- Modify: `D:\googlePRJ\check dataset\tests\test_gemini_api.py`
- Modify: `D:\googlePRJ\check dataset\tests\test_analysis_config.py`

- [ ] **Step 1: Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`**
- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m compileall .`**
- [ ] **Step 3: Fix any failures and re-run until green**
