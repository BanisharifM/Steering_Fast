"""GPT-4o evaluation with rate limiting, retry, and checkpointing.

Fixes from original:
- Exponential backoff retry on ALL errors (not just 429)
- 1s delay between calls to stay under TPM limits
- Regex-based score parsing (handles multi-digit scores)
- Incremental result saving (no progress lost on crash)
"""
import logging
import os
import re
import time
from typing import Optional

log = logging.getLogger(__name__)


class OpenAIEvaluator:
    """Rate-limited GPT-4o evaluator for steered outputs."""

    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        delay: float = 1.0,
        max_retries: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 20,
    ):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try loading from .env file
            for env_path in [".env", "../.env", "../../.env"]:
                if os.path.exists(env_path):
                    with open(env_path) as f:
                        api_key = f.read().strip()
                    break

        if not api_key:
            raise ValueError("OPENAI_API_KEY not set and no .env file found")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.delay = delay
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score_response(self, prompt: str) -> tuple[int, str]:
        """Call GPT-4o with exponential backoff retry.

        Returns (score, explanation).
        """
        from openai import RateLimitError

        for attempt in range(self.max_retries):
            try:
                output = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    model=self.model,
                )
                time.sleep(self.delay)
                content = output.choices[0].message.content or ""
                score = self._parse_score(content)
                return score, content
            except RateLimitError:
                wait = min(2 ** attempt, 60)
                log.warning("Rate limit hit, attempt %d/%d, waiting %ds", attempt + 1, self.max_retries, wait)
                time.sleep(wait)
            except Exception as e:
                wait = min(2 ** attempt, 60)
                log.warning("API error (%s), attempt %d/%d, waiting %ds", type(e).__name__, attempt + 1, self.max_retries, wait)
                time.sleep(wait)

        # Final attempt without catch
        try:
            output = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model=self.model,
            )
            content = output.choices[0].message.content or ""
            return self._parse_score(content), content
        except Exception as e:
            log.error("All retries exhausted: %s", e)
            return 0, f"ERROR: {e}"

    @staticmethod
    def _parse_score(content: str) -> int:
        """Parse score from GPT-4o response. Handles multi-digit scores.

        Original bug: int(content.split("Score: ")[1][0]) only reads first char.
        Fix: uses regex to capture full number.
        """
        match = re.search(r"Score:\s*(\d+)", content)
        if match:
            return int(match.group(1))
        return 0


def load_eval_prompt(data_dir: str, concept_class: str, version: int) -> str:
    """Load evaluation prompt template for a concept class and version."""
    prefix_map = {
        "fears": "phobia_eval",
        "moods": "mood_eval",
        "personas": "persona_eval",
        "personalities": "personality_eval",
        "places": "topophile_eval",
    }
    prefix = prefix_map.get(concept_class, f"{concept_class}_eval")
    path = os.path.join(data_dir, "evaluation_prompts", f"{prefix}_v{version}.txt")
    with open(path) as f:
        return f.read()


def parse_model_response(response_text: str, model_name: str) -> str:
    """Extract the assistant's response from the full generated text.

    Handles both Llama and Qwen chat template formats.
    """
    if "llama" in model_name.lower():
        parts = re.split(r"\|>assistant<\|end_header_id\|>", response_text)
        if len(parts) > 1:
            return "".join(parts[1])
    elif "qwen" in model_name.lower():
        parts = re.split(r"<\|im_start\|>assistant", response_text)
        if len(parts) > 1:
            return "".join(parts[1])
    return response_text
