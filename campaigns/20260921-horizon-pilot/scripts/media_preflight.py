"""CPU-only offline dependency preflight. Does not launch Isaac or a GPU kernel."""
import json
import lerobot.policies.smolvla.configuration_smolvla
from lerobot.policies.factory import make_pre_post_processors
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

model = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
config = AutoConfig.from_pretrained(model, local_files_only=True)
processor = AutoProcessor.from_pretrained(model, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
print(json.dumps({"offline_cache": "available", "model": model,
                  "config_type": type(config).__name__,
                  "processor_type": type(processor).__name__,
                  "tokenizer_type": type(tokenizer).__name__,
                  "factory": callable(make_pre_post_processors)}))
