# Project Development Rules

1.  **Unified Speech Service:** Always utilize the unified speech service defined by the `SPEECH_SERVICE_URL` in the `.env` file for both transcription (STT) and text-to-speech (TTS) functionalities.
2.  **Database Interaction:** All database operations must interact with the central SQLite database `backend/healthcare.db`, preferably through the existing utility functions (`database_utils.py`, `database.py`).
3.  **Conciseness & Efficiency:** Prioritize writing concise and efficient Python code. Leverage modern idioms, built-in functions, list/dict comprehensions, and functional patterns where appropriate. Avoid unnecessary verbosity.
4.  **DRY Principle:** Strictly adhere to the "Don't Repeat Yourself" principle. Refactor redundant code into reusable functions or modules.
5.  **Data Structures:** Choose the most appropriate Python data structures (lists, dicts, sets, tuples) to simplify logic and enhance performance.
6.  **No New Tests:** Do not create new test files. Focus on implementing required features.
7.  **Focus:** Implement only the agreed-upon tasks. Seek clarification before adding unrequested features or making assumptions.
8.  **Respect Boundaries:** Do not modify any files related to `*_server` or `*_client` components, or the core `backend/api.py` file, unless explicitly instructed and confirmed. These seem to be managed by other developers. 