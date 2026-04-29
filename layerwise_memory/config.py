"""
Configuration for Hierarchical Memory System

Handles API keys, model selection, and environment detection.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
USE_MOCK = os.getenv('USE_MOCK_LLM', 'false').lower() == 'true'
API_KEY = (os.getenv('MODEL_API_KEY')
           or os.getenv('OPENROUTER_API_KEY')
           or os.getenv('OPENAI_API_KEY')
           or '')
BASE_URL = "https://openrouter.ai/api/v1"

# Model configuration
LLM_MODEL = "openai/gpt-oss-120b:free"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_DIMENSIONS = 2048

# Merge thresholds (can be overridden in method calls)
DEFAULT_CONCEPT_MERGE_THRESHOLD = 0.75
DEFAULT_PROCESS_MERGE_THRESHOLD = 0.75
DEFAULT_REFLECTION_MERGE_THRESHOLD = 0.8

# Prompt mode configuration
# Options: "conversational" (default) or "medical"
PROMPT_MODE = os.getenv('PROMPT_MODE', 'medical').lower()

OPENAI_AVAILABLE = True  # Assumed; error surfaces on first use if not installed


class _LazyAsyncClient:
    """AsyncOpenAI proxy — defers the openai import to first use."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._real = None

    def _get(self):
        if self._real is None:
            from openai import AsyncOpenAI
            self._real = AsyncOpenAI(**self._kwargs)
        return self._real

    def __getattr__(self, name):
        return getattr(self._get(), name)


class _LazySyncClient:
    """OpenAI proxy — defers the openai import to first use."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._real = None

    def _get(self):
        if self._real is None:
            from openai import OpenAI
            self._real = OpenAI(**self._kwargs)
        return self._real

    def __getattr__(self, name):
        return getattr(self._get(), name)


# Initialize clients (lazy — openai is imported only on first LLM/embedding call)
client = None
async_client = None
if not USE_MOCK:
    if not API_KEY:
        print("WARNING: MODEL_API_KEY not set. Falling back to mock implementations.", file=sys.stderr)
        print("Set USE_MOCK_LLM=true to suppress this warning.", file=sys.stderr)
        USE_MOCK = True
    else:
        client = _LazySyncClient(api_key=API_KEY, base_url=BASE_URL, max_retries=6)
        async_client = _LazyAsyncClient(api_key=API_KEY, base_url=BASE_URL, max_retries=6)

# Export configuration
__all__ = [
    'USE_MOCK',
    'API_KEY',
    'LLM_MODEL',
    'EMBEDDING_MODEL',
    'EMBEDDING_DIMENSIONS',
    'DEFAULT_CONCEPT_MERGE_THRESHOLD',
    'DEFAULT_PROCESS_MERGE_THRESHOLD',
    'DEFAULT_REFLECTION_MERGE_THRESHOLD',
    'PROMPT_MODE',
    'client',
    'async_client',
    'OPENAI_AVAILABLE',
]
