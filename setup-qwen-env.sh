#!/bin/bash

# Qwen Code Environment Setup
# Source this script to configure environment variables for qwen CLI

export QWEN_CODE_OPENAI_API_BASE="http://localhost:8000/v1"
export QWEN_CODE_OPENAI_API_KEY="LOCAL-DEV"

echo "Environment variables configured:"
echo "QWEN_CODE_OPENAI_API_BASE=$QWEN_CODE_OPENAI_API_BASE"
echo "QWEN_CODE_OPENAI_API_KEY=$QWEN_CODE_OPENAI_API_KEY"
echo ""
echo "To use these permanently, add these lines to your ~/.bashrc:"
echo "export QWEN_CODE_OPENAI_API_BASE=\"http://localhost:8000/v1\""
echo "export QWEN_CODE_OPENAI_API_KEY=\"LOCAL-DEV\"" 