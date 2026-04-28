# Refactoring Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix security issues, eliminate code duplication, improve thread safety, harden error handling, and simplify overly complex functions — without changing external behavior.

**Architecture:** Extract shared analysis logic from `standard.py` and `smart.py` into a common module `analysis/common.py`. Add threading locks to all global state mutations in `core/state.py`. Replace bare `except:` clauses with specific exception types. Keep the Gradio UI and API integrations unchanged.

**Tech Stack:** Python, Gradio, unittest, existing modules

**Priority Legend:** P0 = critical/security, P1 = high, P2 = medium, P3 = low

---

## ~~Chunk 1: Security — credential rotation and git history cleanup (P0)~~

> **SKIPPED:** `.env` and `canvas-genius-*.json` exist on disk but are correctly excluded by `.gitignore` and were never committed to git history. No credential leak, no rotation needed.

---

## Chunk 2: Thread safety for global state (P0)

### Task 2: Add locks to all global state mutations

**Files:**
- Modify: `core/state.py`
- Create: `tests/test_state_thread_safety.py`

- [ ] **Step 1: Write failing tests**

```python
import threading
from core.state import set_global_results, get_global_results

def test_concurrent_set_global_results_no_corruption():
    """Two threads writing results simultaneously must not lose data."""
    errors = []
    def writer(value):
        try:
            set_global_results(value)
            result = get_global_results()
            assert isinstance(result, list)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer, args=([{"id": 1}],))
    t2 = threading.Thread(target=writer, args=([{"id": 2}],))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert not errors
```

- [ ] **Step 2: Run tests and verify they pass (baseline — race conditions are non-deterministic, so these establish the API)**
- [ ] **Step 3: Add `threading.Lock()` to every global variable in `state.py`**

Wrap each getter/setter pair:

```python
_results_lock = threading.Lock()
_cache_lock = threading.Lock()

def set_global_results(results):
    with _results_lock:
        global global_results
        global_results = results

def get_global_results():
    with _results_lock:
        return global_results
```

Apply to: `global_results`, `dataset_cache`, `current_dataset_name`, `current_dataset_config`, `verified_indices`, `discarded_indices`.

- [ ] **Step 4: Run full test suite, verify no regressions**
- [ ] **Step 5: Commit**

---

## Chunk 3: Extract shared analysis logic to eliminate duplication (P1)

### Task 3: Create `analysis/common.py` with shared helpers

**Files:**
- Create: `analysis/common.py`
- Modify: `analysis/standard.py`
- Modify: `analysis/smart.py`
- Create: `tests/test_analysis_common.py`

- [ ] **Step 1: Identify duplicated blocks**

Target functions with near-identical logic in both modules:

| standard.py | smart.py | Shared concept |
|---|---|---|
| `_run_fresh_analysis()` lines 275-530 | `_smart_fresh_first_pass()` lines 100-310 | Fresh analysis loop: load dataset, iterate files, call transcriber, compute similarity, update results |
| `_run_recheck_analysis()` lines 532-750 | `_smart_recheck_first_pass()` lines 312-440 | Recheck loop: filter problematic results, re-transcribe, update |
| Audio loading + fallback logic (multiple sites) | Same pattern | `resolve_audio(ds, result, path_index)` |
| Result initialization + model_results merge | Same pattern | `init_result_entry(...)`, `merge_model_result(...)` |

- [ ] **Step 2: Write tests for the extracted helpers**

```python
from analysis.common import resolve_audio, init_result_entry, merge_model_result

def test_init_result_entry_has_required_fields():
    entry = init_result_entry(file_path="audio/001.wav", reference_text="hello")
    assert entry["file"] == "audio/001.wav"
    assert entry["reference_text"] == "hello"
    assert entry["model_results"] == {}
    assert entry["status"] == "pending"

def test_merge_model_result_preserves_existing():
    entry = init_result_entry("a.wav", "text")
    merge_model_result(entry, model_name="flash", transcription="text", similarity=95.0)
    merge_model_result(entry, model_name="pro", transcription="text!", similarity=90.0)
    assert "flash" in entry["model_results"]
    assert "pro" in entry["model_results"]

def test_resolve_audio_returns_none_for_missing():
    audio, sr = resolve_audio(ds=None, result={}, path_index={})
    assert audio is None
```

- [ ] **Step 3: Run tests, verify they fail**
- [ ] **Step 4: Implement `analysis/common.py`**

Extract into `common.py`:
- `resolve_audio(ds, result, path_index) -> (audio_array, sampling_rate)`
- `init_result_entry(file_path, reference_text, **kwargs) -> dict`
- `merge_model_result(entry, model_name, transcription, similarity, raw_response=None)`
- `build_path_index(dataset) -> dict`
- `filter_by_scope(results, scope) -> list[int]` (indices to process)
- `has_valid_audio(audio_data) -> bool` (replaces `_has_audio_data` with proper numpy checks)

- [ ] **Step 5: Run tests, verify they pass**
- [ ] **Step 6: Replace duplicated code in `standard.py` with calls to `common.py`**
- [ ] **Step 7: Replace duplicated code in `smart.py` with calls to `common.py`**
- [ ] **Step 8: Run full test suite, verify no regressions**
- [ ] **Step 9: Commit**

---

## Chunk 4: Replace bare `except:` with specific exception handling (P1)

### Task 4: Fix silent error swallowing

**Files:**
- Modify: `analysis/smart.py` (lines ~280, ~396)
- Modify: `analysis/import_export.py` (line ~130)
- Modify: `hf_asr.py` (line ~207)
- Modify: `tests/test_analysis_logs.py`

- [ ] **Step 1: Write tests that verify errors are logged, not swallowed**

```python
def test_smart_analysis_logs_transcription_error(capfd):
    """When transcription raises, error should appear in analysis logs."""
    # ... mock GeminiIntegrator.transcribe_audio to raise RuntimeError
    # ... run smart analysis on a single file
    # ... assert "RuntimeError" or similar appears in get_analysis_logs()
```

- [ ] **Step 2: Run tests, verify they fail**
- [ ] **Step 3: Replace each bare `except:` with specific types and logging**

Changes:

```python
# BEFORE (smart.py:280)
except:
    pass

# AFTER
except (RuntimeError, ValueError, OSError) as exc:
    add_analysis_log(f"Transcription error for {file_path}: {exc}")
```

```python
# BEFORE (import_export.py:130)
except:
    model_results = {}

# AFTER
except (json.JSONDecodeError, TypeError, ValueError):
    model_results = {}
```

```python
# BEFORE (hf_asr.py:207)
except:
    pass

# AFTER
except OSError:
    pass  # temp file already cleaned up
```

- [ ] **Step 4: Run tests, verify they pass**
- [ ] **Step 5: Run full test suite**
- [ ] **Step 6: Commit**

---

## Chunk 5: Fix unsafe type conversions (P1)

### Task 5: Guard `int(float(...))` conversions in CSV import

**Files:**
- Modify: `analysis/import_export.py`
- Modify: `tests/test_analysis_config.py`

- [ ] **Step 1: Write failing test**

```python
def test_import_csv_handles_non_numeric_id_gracefully():
    """A row with id='abc' should be skipped, not crash."""
    # Create CSV with a non-numeric id column value
    # Call import_csv_analysis(...)
    # Assert no exception raised, row is skipped
```

- [ ] **Step 2: Run test, verify it fails**
- [ ] **Step 3: Wrap conversion in try/except**

```python
# BEFORE (import_export.py:106)
if pd.notnull(row_id) and str(row_id).strip() != '' and (int_id := int(float(row_id))) in id_to_idx:

# AFTER
if pd.notnull(row_id) and str(row_id).strip() != '':
    try:
        int_id = int(float(row_id))
    except (ValueError, OverflowError):
        continue
    if int_id in id_to_idx:
```

- [ ] **Step 4: Run tests, verify they pass**
- [ ] **Step 5: Commit**

---

## Chunk 6: Simplify complex functions (P2)

### Task 6: Break down `run_analysis()` in standard.py

**Files:**
- Modify: `analysis/standard.py`
- Modify: `tests/test_analysis_config.py`

- [ ] **Step 1: Map current `run_analysis()` responsibilities**

Current flow (lines 166-274):
1. Parse and validate parameters (model, exec mode, scope)
2. Decide which sub-function to call (fresh/recheck/batch/hf)
3. Call sub-function
4. Post-process results (save CSV, update state)
5. Return dashboard HTML

- [ ] **Step 2: Extract parameter validation into `_validate_analysis_params()`**

```python
def _validate_analysis_params(dataset_name, model_name, exec_mode, scope):
    """Return validated (dataset_name, model_name, exec_mode, scope) or raise ValueError."""
```

- [ ] **Step 3: Extract post-processing into `_finalize_analysis(results, csv_path)`**

```python
def _finalize_analysis(results, csv_path=None):
    """Save CSV, update global state, return dashboard HTML."""
```

- [ ] **Step 4: Simplify `run_analysis()` to ~30 lines of orchestration**
- [ ] **Step 5: Run full test suite**
- [ ] **Step 6: Commit**

### Task 7: Break down `_run_hf_fresh_analysis()` (240 lines -> ~3 functions)

**Files:**
- Modify: `analysis/standard.py`

- [ ] **Step 1: Extract batch submission loop into `_submit_hf_batch()`**
- [ ] **Step 2: Extract result collection loop into `_collect_hf_results()`**
- [ ] **Step 3: Keep `_run_hf_fresh_analysis()` as thin orchestrator**
- [ ] **Step 4: Run full test suite**
- [ ] **Step 5: Commit**

### Task 8: Break down `import_csv_analysis()` (144 lines -> ~3 functions)

**Files:**
- Modify: `analysis/import_export.py`

- [ ] **Step 1: Extract column mapping into `_map_csv_columns(df) -> df`**
- [ ] **Step 2: Extract row matching into `_match_rows_to_dataset(df, dataset) -> matches`**
- [ ] **Step 3: Keep `import_csv_analysis()` as orchestrator**
- [ ] **Step 4: Run full test suite**
- [ ] **Step 5: Commit**

---

## Chunk 7: Improve audio validation (P2)

### Task 9: Fix `_has_audio_data()` for edge cases

**Files:**
- Modify: `analysis/standard.py` (or move to `analysis/common.py` if Chunk 3 is done)
- Modify: `tests/test_ui_audio.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np

def test_has_valid_audio_rejects_empty_2d():
    assert not has_valid_audio(np.array([[]]))

def test_has_valid_audio_rejects_scalar():
    assert not has_valid_audio(np.float32(0.0))

def test_has_valid_audio_rejects_dict():
    assert not has_valid_audio({"array": [1, 2, 3]})

def test_has_valid_audio_accepts_valid_1d():
    assert has_valid_audio(np.array([0.1, 0.2, 0.3]))
```

- [ ] **Step 2: Run tests, verify they fail**
- [ ] **Step 3: Implement robust check**

```python
def has_valid_audio(audio_data) -> bool:
    if audio_data is None:
        return False
    if not isinstance(audio_data, np.ndarray):
        return False
    if audio_data.ndim == 0 or audio_data.size == 0:
        return False
    return True
```

- [ ] **Step 4: Replace all `_has_audio_data()` call sites**
- [ ] **Step 5: Run full test suite**
- [ ] **Step 6: Commit**

---

## Chunk 8: Remove global print hijacking (P2)

### Task 10: Replace `builtins.print` override with explicit logging

**Files:**
- Modify: `core/state.py`
- Modify: `app.py`
- Modify: any files that rely on hijacked `print()` for log capture

- [ ] **Step 1: Grep for all `print(` calls across the project to understand scope**
- [ ] **Step 2: Introduce a thin `log(msg)` function in `core/state.py` that writes to `analysis_logs`**
- [ ] **Step 3: Replace `print()` calls in analysis modules with `log()`**
- [ ] **Step 4: Remove `install_log_capture()` and the `builtins.print` override**
- [ ] **Step 5: Run full test suite**
- [ ] **Step 6: Commit**

---

## Chunk 9: Pin dependency versions (P3)

### Task 11: Pin all dependencies in requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Run `pip freeze` in the venv to capture current working versions**
- [ ] **Step 2: Update `requirements.txt` with pinned versions (e.g. `gradio==5.x.x`)**
- [ ] **Step 3: Verify `pip install -r requirements.txt` succeeds in a fresh venv**
- [ ] **Step 4: Commit**

---

## Execution order and dependencies

```
Chunk 1 (Security)          -- SKIPPED, no leak found
Chunk 2 (Thread safety)     -- no dependencies, start here
Chunk 3 (Extract common)    -- no dependencies, but do before Chunks 6-7
Chunk 4 (Error handling)    -- no dependencies
Chunk 5 (Type safety)       -- no dependencies
  ---- above are P0/P1, do first ----
Chunk 6 (Simplify funcs)    -- depends on Chunk 3 (uses common.py)
Chunk 7 (Audio validation)  -- depends on Chunk 3 (lives in common.py)
Chunk 8 (Remove print hack) -- no dependencies
Chunk 9 (Pin deps)          -- do last
```

Chunks 2, 3, 4, 5 are independent and can be worked on in parallel after Chunk 1.

---

## Success criteria

- [x] No secrets in git history (verified: never committed)
- [x] All global state mutations use locks
- [x] Zero bare `except:` in codebase (`grep -r "except:" --include="*.py" | grep -v "except .* as"` returns nothing)
- [ ] No function exceeds 80 lines (excluding tests)
- [x] Duplicated analysis code reduced (common.py extracted, HF batch loop deduplicated)
- [x] Full test suite passes with no new warnings
- [x] `has_valid_audio()` rejects scalar, dict, empty-2d numpy arrays
