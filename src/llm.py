import asyncio
import logging

from litellm import acompletion
from litellm.exceptions import RateLimitError, BadRequestError

logger = logging.getLogger(__name__)

MODELS = [
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/google/gemini-2.5-flash",
]


class LLMClient:
    """Rotates through models on rate limit errors, cycling until one responds."""

    def __init__(self, models: list[str] | None = None):
        self.models = models or list(MODELS)

    async def call(
        self,
        messages: list[dict],
        temperature: float = 0.0,
    ) -> str:
        while True:
            for model in self.models:
                try:
                    response = await acompletion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    )
                    logger.debug("LLM responded: model=%s", model)
                    return response.choices[0].message.content
                except (RateLimitError, BadRequestError) as e:
                    logger.warning("Model %s failed: %s", model, str(e)[:150])
            logger.info("All models rate-limited, waiting 10s...")
            await asyncio.sleep(10)
