# Qwen Code CLI Setup Guide

## Current Status ✅

Your system is configured with:
- ✅ Python 3.11.13 in `qwen` conda environment
- ✅ PyTorch 2.7.1 with CUDA 12.8 support  
- ✅ vLLM 0.10.0 (supports GGUF models)
- ✅ Transformers 4.54.1
- ✅ Node.js v20.19.4
- ✅ Qwen CLI installed (command: `qwen`)

## Model Download

**Currently downloading:** Qwen3-Coder-30B-A3B-Instruct (~18GB)
- **Performance:** BFCL-v3 69.1 · LiveCodeBench-v5 62.6 · CodeForces 1977
- **Memory usage:** ~18GB + ~7GB KV cache = ~25GB total (fits RTX 5090)

Monitor download progress:
```bash
watch -n 5 'du -sh $HOME/models/qwen3-30b-a3b-4bit/ 2>/dev/null || echo "Still downloading..."'
```

## Setup Instructions

### 1. Environment Setup

Source the environment script:
```bash
source setup-qwen-env.sh
```

Or set manually each time:
```bash
export QWEN_CODE_OPENAI_API_BASE="http://localhost:8000/v1"
export QWEN_CODE_OPENAI_API_KEY="LOCAL-DEV"
```

### 2. Start the vLLM Server

Once the model download completes, run:
```bash
./setup-qwen-server.sh
```

This will:
- Check if the model is ready
- Activate the qwen conda environment
- Start vLLM server on http://localhost:8000/v1
- Enable reasoning mode for agentic behavior

### 3. Using the CLI

Once the server is running, in a new terminal:

```bash
# Activate environment
conda activate qwen
source setup-qwen-env.sh

# Initialize a new project
qwen init myproject

# Start interactive coding session
qwen

# Run with a specific prompt
qwen -p "Add unit tests to this codebase"

# Auto-accept all changes (YOLO mode)
qwen -y -p "Refactor this code for better performance"
```

## CLI Options Reference

- `-m, --model` - Model name (default: "qwen3-coder-plus")
- `-p, --prompt` - Non-interactive prompt
- `-i, --prompt-interactive` - Execute prompt and continue interactively
- `-s, --sandbox` - Run in sandbox mode
- `-a, --all-files` - Include ALL files in context
- `-y, --yolo` - Auto-accept all actions
- `-d, --debug` - Debug mode
- `--openai-api-key` - Override API key
- `--openai-base-url` - Override base URL

## Alternative Models

If you want a smaller model, you can download these instead:

### Qwen3-14B-Instruct (~11GB)
```bash
huggingface-cli download avoroshilov/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g \
  --local-dir $HOME/models/qwen3-14b-4bit
```

### Qwen2.5-7B-Instruct (~6GB)  
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  --local-dir $HOME/models/qwen25-7b-gguf --include '*.gguf'
```

Then update `setup-qwen-server.sh` to point to the desired model path.

## Troubleshooting

### Server Issues
- **Port conflict:** Change `--port 8000` to another port in setup-qwen-server.sh
- **Memory issues:** Try a smaller model or reduce `--max-model-len`
- **CUDA errors:** Check `nvidia-smi` and ensure drivers are working

### CLI Issues
- **Command not found:** Ensure `/usr/bin` is in your PATH
- **Connection failed:** Check if vLLM server is running on localhost:8000
- **Environment variables:** Re-source setup-qwen-env.sh

### Performance Tuning
- **Longer contexts:** Add `--rope-scaling` for >32K tokens
- **Multi-GPU:** Increase `--tensor-parallel-size` 
- **Faster inference:** Disable reasoning with `--no-think` flag

## GPU Monitoring

Monitor GPU usage during inference:
```bash
watch -n 1 nvidia-smi
```

Expected usage:
- **Weights:** ~18GB VRAM
- **KV Cache:** ~7GB at 8K context
- **Total:** ~25GB (fits comfortably on RTX 5090)

## Next Steps

Once setup is complete:
1. ✅ Start the vLLM server 
2. ✅ Test basic CLI functionality
3. ✅ Try an interactive coding session
4. ✅ Experiment with different prompts and modes 