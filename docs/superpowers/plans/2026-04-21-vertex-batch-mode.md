# Vertex Batch Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real Vertex Batch API path that uploads audio to GCS, submits all required batch jobs up front in chunks of 100 files, waits for completion, and exposes a UI selector for direct, flex, and batch execution modes.

**Architecture:** Keep the existing direct inline Vertex path intact, add batch-only helpers in `gemini_api.py`, and branch from standard analysis based on the new execution mode selector. Reuse the current result schema so dashboard/export code does not need structural changes.

**Tech Stack:** Python, Gradio, `google-genai`, Google Cloud Storage, Vertex AI batch jobs, `unittest`

---

### Task 1: Add failing tests for batch helpers and mode selection

**Files:**
- Modify: `D:\googlePRJ\check dataset\tests\test_gemini_api.py`
- Modify: `D:\googlePRJ\check dataset\tests\test_analysis_config.py`

- [ ] **Step 1: Write failing tests for chunking, JSONL request generation, and output parsing**
- [ ] **Step 2: Run targeted tests and confirm they fail for missing batch functionality**
- [ ] **Step 3: Write failing tests for execution mode config/validation**
- [ ] **Step 4: Re-run targeted tests and confirm failures are for the new missing behavior**

### Task 2: Implement Vertex batch helpers

**Files:**
- Modify: `D:\googlePRJ\check dataset\gemini_api.py`
- Modify: `D:\googlePRJ\check dataset\requirements.txt`

- [ ] **Step 1: Add batch-mode constants, config validation, and GCS path helpers**
- [ ] **Step 2: Add helpers to upload WAV audio to GCS and generate JSONL request payloads with `fileData.fileUri`**
- [ ] **Step 3: Add helpers to create all Vertex batch jobs up front, poll them, and collect output JSONL**
- [ ] **Step 4: Re-run batch helper tests and make them pass**

### Task 3: Integrate batch mode into standard analysis

**Files:**
- Modify: `D:\googlePRJ\check dataset\analysis\standard.py`

- [ ] **Step 1: Add execution mode branching without disturbing HF ASR flow**
- [ ] **Step 2: Implement fresh-analysis batch path that preloads dataset items, stages all jobs, and maps outputs back into existing result records**
- [ ] **Step 3: Keep recheck on direct path unless a safe batch recheck slice is straightforward**
- [ ] **Step 4: Run analysis config tests and new integration-targeted tests**

### Task 4: Expose execution mode in UI

**Files:**
- Modify: `D:\googlePRJ\check dataset\ui\gradio_app.py`

- [ ] **Step 1: Replace the flex checkbox with a direct/flex/batch selector**
- [ ] **Step 2: Map selector values into analysis inputs while preserving smart analysis behavior**
- [ ] **Step 3: Update UI copy so each mode’s behavior is explicit**

### Task 5: Final verification

**Files:**
- Modify: `D:\googlePRJ\check dataset\README.md`

- [ ] **Step 1: Document required GCS env vars for batch staging**
- [ ] **Step 2: Run the targeted unittest suite**
- [ ] **Step 3: Fix any failures and re-run until green**
