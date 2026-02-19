#!/bin/bash
# Post-edit hook: validate JSON syntax for spot config files
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only validate JSON files in data/spots/
if [[ "$FILE_PATH" != */data/spots/*.json ]]; then
  exit 0
fi

# Validate JSON syntax
if ! jq empty "$FILE_PATH" 2>/dev/null; then
  echo "Invalid JSON syntax in $FILE_PATH" >&2
  exit 2
fi

exit 0
