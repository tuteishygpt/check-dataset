# Analysis Scope Selector Design

**Feature:** selector for choosing which files are sent to analysis from the frontend.

**Goal:** replace the boolean "recheck only problematic files" toggle with four explicit modes and keep the backend selection logic consistent across standard, smart, and HF analysis.

## Decisions

1. Replace the checkbox in `ui/gradio_app.py` with a selector that exposes:
   - `all`
   - `problematic`
   - `pending` (default)
   - `problematic_or_pending`
2. Pass the selector value through the UI handlers into `analysis.standard.run_analysis()` and `analysis.smart.run_smart_analysis()`.
3. Centralize record filtering in shared helpers inside `analysis.standard` so the same semantics apply to:
   - Gemini standard analysis
   - HF ASR analysis
   - smart analysis
4. Preserve the current batch restriction:
   - batch mode still runs only for fresh/full analysis
   - non-`all` scopes fall back to direct processing with a warning

## Selection Semantics

- `all`: analyze the whole dataset flow exactly as a fresh run does today.
- `problematic`: analyze only records with score below the threshold and not marked `correct`.
- `pending`: analyze only records with `verification_status == "pending"`.
- `problematic_or_pending`: analyze the union of `problematic` and `pending` without duplicates, preserving original order.

## Testing

- Add unit tests for scope selection helper behavior.
- Add routing tests for standard analysis batch fallback with non-`all` scope values.
- Keep existing tests passing by updating old boolean-based calls to the new scope API.
