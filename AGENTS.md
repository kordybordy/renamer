# Agent Guidelines


This file is **authoritative** and must be read by the AI assistant **before starting any task**
(fixing bugs, adding features, refactoring, or making improvements).

Its purpose is to ensure the AI **starts in the correct place**, applies **minimal safe changes**,
and respects the project’s constraints.

---

## 1. Scope

- **Scope:** entire repository
- All files are in scope, but changes should be **targeted and minimal**
- The application is a **PDF renaming tool** using AI (OpenAI or Ollama) to generate filenames

---

## 2. Repository Structure (Source of Truth)
.
├── main.py # Primary application logic AND GUI
├── assets/ # Icons and static resources bundled with the app
├── renamer.spec # Packaging configuration (PyInstaller)
└── build.bat # Windows build helper

### Key implication for AI
- **`main.py` is the central entry point**
- There is no deep modular structure — always start by reading `main.py`
- Do NOT invent new directories or split files unless explicitly asked

---

## 3. Where to Start for Any Task

### Default rule
➡️ **Open `main.py` first, locate the relevant section, and work locally**

### Task-specific guidance

#### A. Bug fixes
- Identify the smallest code region responsible
- Apply **surgical fixes**, not refactors
- Add guards instead of restructuring logic

#### B. New features
- Integrate into existing flow in `main.py`
- Reuse existing variables, patterns, and UI elements
- Avoid introducing new global state unless unavoidable

#### C. AI-related changes (OpenAI / Ollama)
- Modify only:
  - prompt construction
  - model selection logic
  - AI call handling
- Do NOT change GUI or file-handling code unless required

#### D. GUI-related changes
- Be extremely careful with signal handlers
- Assume handlers **may be invoked early**
- Guards (`if not ready: return`) are welcome and encouraged

---

## 4. GUI Safety Rules (Very Important)

- GUI signal handlers must be **safe against early or repeated invocation**
- Prefer defensive checks over assumptions
- Never assume files, folders, or AI results are already available

Example philosophy:
- ✅ guard and return
- ❌ crash or block the UI

---

## 5. Coding Style Expectations

- Prefer **small, focused changes**
- Match existing naming and formatting
- Avoid large refactors unless explicitly requested
- Do not rename public functions casually
- Do not redesign the architecture

If unsure:
➡️ extend existing code instead of creating new abstractions

---

## 6. Error Handling Philosophy

The app should:
- Skip problematic files
- Log or surface errors clearly
- Continue batch processing whenever possible

Prefer:
- graceful failure
- clear logging
over:
- uncaught exceptions
- stopping the entire run

---

## 7. Performance Assumptions

- The app may process many PDFs
- AI calls are slow and expensive

Therefore:
- Avoid duplicate AI calls
- Do not block the GUI unnecessarily
- Reuse extracted text where possible

---

## 8. Build & Packaging Awareness

- `renamer.spec` and `build.bat` are for distribution builds
- Do NOT modify them unless the task explicitly involves packaging
- Assume assets in `assets/` must remain compatible with PyInstaller

---

## 9. Mandatory Pre-Commit Check

Before considering a task complete, the AI **must ensure**:

```bash
python -m compileall main.py

Code must compile without errors
Do not ignore syntax warnings or failures
