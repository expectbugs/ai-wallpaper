#!/usr/bin/env python3
import torch
import sys
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    from diffusers import FluxPipeline
    print("FluxPipeline imported successfully")
except Exception as e:
    print(f"ERROR importing FluxPipeline: {e}")
    sys.exit(1)

try:
    print("Testing basic FLUX loading...")
    model_path = "/home/user/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    print("FLUX model loaded successfully!")
except Exception as e:
    print(f"ERROR loading FLUX: {e}")
    import traceback
    traceback.print_exc()