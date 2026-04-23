# Analysis Scope Selector Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replace the frontend recheck checkbox with a four-state scope selector and make every analysis path process only the files selected by that scope.

**Architecture:** Add one shared selection helper in `analysis.standard`, thread a string scope value through the Gradio UI into standard and smart analysis entry points, and reuse the helper in Gemini, HF, and smart flows. Preserve existing fresh-analysis behavior for `all` and existing resume behavior for `pending`, but make the scope explicit instead of inferred from a boolean.

**Tech Stack:** Python, Gradio, unittest, existing analysis modules

---

## Chunk 1: Shared scope selection

### Task 1: Define scope constants and helper tests

**Files:**
- Modify: `tests/test_analysis_config.py`
- Modify: `analysis/standard.py`

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run the targeted tests and verify they fail**
- [ ] **Step 3: Implement shared scope helpers with minimal code**
- [ ] **Step 4: Run the targeted tests and verify they pass**
- [ ] **Step 5: Commit**

### Task 2: Route standard and HF analysis through explicit scopes

**Files:**
- Modify: `analysis/standard.py`
- Modify: `tests/test_analysis_config.py`

- [ ] **Step 1: Write failing routing tests for non-`all` scope behavior**
- [ ] **Step 2: Run the targeted tests and verify they fail**
- [ ] **Step 3: Implement minimal routing changes**
- [ ] **Step 4: Run the targeted tests and verify they pass**
- [ ] **Step 5: Commit**

## Chunk 2: Smart analysis and frontend wiring

### Task 3: Reuse scope selection in smart analysis

**Files:**
- Modify: `analysis/smart.py`
- Modify: `tests/test_analysis_config.py`

- [ ] **Step 1: Write failing tests for smart scope forwarding if needed**
- [ ] **Step 2: Run the targeted tests and verify they fail**
- [ ] **Step 3: Implement minimal smart-analysis changes**
- [ ] **Step 4: Run the targeted tests and verify they pass**
- [ ] **Step 5: Commit**

### Task 4: Replace the frontend checkbox with a selector

**Files:**
- Modify: `ui/gradio_app.py`

- [ ] **Step 1: Update the UI control and labels**
- [ ] **Step 2: Thread the selected scope into both analysis handlers**
- [ ] **Step 3: Run targeted tests and a quick smoke verification**
- [ ] **Step 4: Commit**

Plan complete and saved to `docs/superpowers/plans/2026-04-22-analysis-scope-selector.md`. Ready to execute.
