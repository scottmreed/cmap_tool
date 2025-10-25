# Concept Map Builder

Interactive Streamlit + python-pptx tool for authoring concept maps, previewing their layout, and exporting polished PPTX slides.

## Supported Python Versions

| Platform | Versions | Notes |
| --- | --- | --- |
| Windows | 3.9 – 3.12 | `run_app.bat` creates `.venv_win_pyXY` inside the repo and launches Streamlit. |
| macOS/Linux | 3.9 – 3.12 | `run_app.sh` mirrors the Windows workflow via `.venv_unix_pyXY`. |

Python 3.13 is currently blocked because `matplotlib` and `lxml` do not yet provide wheels for that interpreter; attempting to use 3.13 leads to build failures.

## Running the App

### Windows

1. Install Python 3.9–3.12 from [python.org](https://www.python.org/downloads/windows/) and tick **Add Python to PATH**.
2. Double-click `run_app.bat` (or run it from Command Prompt/PowerShell). The script will:
   - detect your Python version,
   - create or reuse `.venv_win_pyXY` within the repository,
   - upgrade `pip`, install `requirements.txt`, and run `streamlit run app.py`.
3. Press `CTRL+C` to stop Streamlit. Delete `.venv_win_pyXY` if you ever want to rebuild the environment from scratch.

### macOS/Linux

1. Ensure Python 3.9–3.12 is on your PATH (`python3 --version`).
2. Run `./run_app.sh`. The script will:
   - select the available Python binary,
   - create or reuse `.venv_unix_pyXY` inside the repo,
   - upgrade `pip`, install dependencies, and start Streamlit.
3. Exit with `CTRL+C` when finished. Remove `.venv_unix_pyXY` if you prefer a fresh setup next time.

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `ERROR: No suitable Python interpreter found` | Install Python 3.9–3.12 and ensure `python`/`python3` is on PATH before rerunning the launcher. |
| Dependency build errors | Install Microsoft C++ Build Tools (Windows) or your platform’s compiler toolchain, then rerun the launcher. |
| Streamlit cannot open a browser | Copy the terminal URL (usually http://localhost:8501) into your browser manually. |

## Project Structure

- `app.py` – Streamlit UI (concept entry, linking, preview, PPTX download).
- `concept_map.py` – Layout + PPTX generation utilities.
- `run_app.bat` / `run_app.sh` – Cross-platform launchers (Docker first, virtual env fallback).
- `requirements.txt` – Dependency ranges tested with Python 3.9-3.12.

## Contributing

See `Agents.md` for authoring guidelines, including the requirement to keep both launcher scripts up to date when dependencies or runtime behavior changes.
