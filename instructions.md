---
description: "Global Python environment setup instructions for AI agent"
applyTo: "**/*.py"
---

# Python Virtual Environment Policy

- Whenever a Python virtual environment is required, always use [uv](https://github.com/astral-sh/uv) to create and manage the environment.
- Do not use `venv`, `virtualenv`, or `conda` for creating Python virtual environments.
- Example command to create a new environment:
uv venv .venv
- Activate the environment using the appropriate method for the user's operating system.
For MacOS:
source .venv/bin/activate
- Make sure to create a requirements.txt file for any necessary requirements. Don't forget to update this file whenever a new library is used.
- Always use the command uv add -r requirements.txt to add any dependencies to the project. Or, use the command uv add <package-name> to add any new package after updating the requirements.txt file.
- Always use Context7 MCP whenever working with any library.