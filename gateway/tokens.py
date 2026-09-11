"""Prompt token accounting.

The cost bucket charges ``prompt_tokens + max_tokens``, so this estimate is
what separates a cheap chat turn from a slot-hogging generation. Exact counts
come from the model's own tokenizer when available; the heuristic fallback
keeps relative ordering intact, which is all the limiter needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TokenizerConfig


@dataclass
class TokenEstimate:
    prompt_tokens: int
    max_tokens: int
    method: str

    @property
    def cost(self) -> int:
        return self.prompt_tokens + self.max_tokens


class TokenCounter:
    def __init__(self, config: TokenizerConfig) -> None:
        self._config = config
        self._tokenizer: Any | None = None
        self._method = "heuristic"
        if config.use_transformers:
            self._try_load()

    def _try_load(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._config.name)
            self._method = "tokenizer"
        except Exception:
            # Offline box, gated repo, no HF token: degrade, do not crash.
            self._tokenizer = None

    @property
    def method(self) -> str:
        return self._method

    def count_text(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, int(len(text) / self._config.chars_per_token))

    def estimate(self, body: dict[str, Any], ceiling: int, default_max: int) -> TokenEstimate:
        prompt_tokens = 0
        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                prompt_tokens += self._config.per_message_overhead
                prompt_tokens += self.count_text(_message_text(message))
        elif isinstance(prompt := body.get("prompt"), str):
            prompt_tokens += self.count_text(prompt)
        elif isinstance(prompt, list):
            for item in prompt:
                prompt_tokens += self.count_text(item if isinstance(item, str) else str(item))

        requested = body.get("max_tokens")
        if requested is None:
            requested = body.get("max_completion_tokens")
        if not isinstance(requested, int) or requested <= 0:
            # An omitted max_tokens is not a cheap request: the server will
            # generate until the context limit. Charge the default, not zero.
            requested = default_max
        max_tokens = min(requested, ceiling)

        # ``n`` and ``best_of`` multiply generation work for one request.
        multiplier = 1
        for key in ("n", "best_of"):
            value = body.get(key)
            if isinstance(value, int) and value > 1:
                multiplier = max(multiplier, value)

        return TokenEstimate(
            prompt_tokens=max(1, prompt_tokens),
            max_tokens=max_tokens * multiplier,
            method=self._method,
        )


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return str(message)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal-style content parts.
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)
