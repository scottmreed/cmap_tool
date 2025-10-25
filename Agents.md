# Agents Guide

This document spells out expectations for anyone (human or automated) modifying the concept-map tool.

## Keep the launch scripts in sync

1. **`run_app.bat` (Windows) and `run_app.sh` (macOS/Linux) must always mirror each other.**  
   - Any change to dependencies, entry points, ports, env vars, or CLI behavior must be applied to both scripts.  
   - Keep their inline help/comments synchronized so users see consistent instructions regardless of platform.
2. **Test every supported execution path.**  
   - Verify both scripts on the minimum and maximum supported Python versions (currently 3.9–3.12).  
   - Ensure the `.venv_win_pyXY` / `.venv_unix_pyXY` directories still bootstrap correctly after your changes.

## Dependency Hygiene

- `requirements.txt` should express version ranges, not floating pins, so that the batch launcher can support multiple Python installations.
- When adding a dependency, document why it is required in the PR/commit description and ensure both the README and this file stay accurate.

## Documentation Expectations

- Any UX or runtime change that affects setup must be mirrored in `README.md` (quick start instructions for both scripts) and, when relevant, in this file.
- If additional startup helpers are introduced (PowerShell, shell, etc.), document them here along with who owns keeping them current.

Following these guidelines keeps onboarding predictable and prevents the Windows launcher from drifting out of date as the project evolves.
