#!/bin/bash
# Post-edit hook: format Python files with black and isort
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only format Python files
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Run formatters (silently)
if command -v black &> /dev/null; then
  black --quiet "$FILE_PATH" 2>/dev/null
fi

if command -v isort &> /dev/null; then
  isort --quiet "$FILE_PATH" 2>/dev/null
fi

exit 0
