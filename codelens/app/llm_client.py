"""
LLM client with configurable provider.

Design decisions:
- Single function `ask_llm()` hides the provider-specific API behind a
  uniform interface. Downstream code never imports openai/anthropic/gemini
  directly — it always goes through this module.
- temperature=0 everywhere. We want deterministic, grounded answers.
- The system prompt is passed in by the caller (not hardcoded here).
  This module is just the transport layer.
- Phase 1 only supports OpenAI. Anthropic and Gemini are stubbed.

Interview note: This is the "Strategy pattern" — swapping implementations
without changing the calling code. The env var picks the strategy.
"""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


def ask_llm(
    prompt: str,
    system_prompt: str = "",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Send a prompt to the configured LLM and return the response text.

    Args:
        prompt: The user/main prompt.
        system_prompt: Optional system-level instructions.
        provider: LLM provider ("openai", "anthropic", "gemini").
        model: Model name for the provider.
        api_key: API key for the provider.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        The LLM's response text.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "openai":
        return _ask_openai(prompt, system_prompt, model, api_key, temperature, max_tokens)
    elif provider == "anthropic":
        return _ask_anthropic(prompt, system_prompt, model, api_key, temperature, max_tokens)
    elif provider == "gemini":
        return _ask_gemini(prompt, system_prompt, model, api_key, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _ask_openai(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call OpenAI's chat completion API."""
    client = OpenAI(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = response.choices[0].message.content or ""
    logger.debug("OpenAI response (%d chars): %s...", len(text), text[:100])
    return text


def _ask_anthropic(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Anthropic's messages API. (Stub — will be implemented when needed.)"""
    # TODO: Implement in Phase 2+ when user requests Anthropic support
    raise NotImplementedError(
        "Anthropic support is planned but not yet implemented. "
        "Set LLM_PROVIDER=openai in your .env file."
    )


def _ask_gemini(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Google's Gemini API. (Stub — will be implemented when needed.)"""
    # TODO: Implement in Phase 2+ when user requests Gemini support
    raise NotImplementedError(
        "Gemini support is planned but not yet implemented. "
        "Set LLM_PROVIDER=openai in your .env file."
    )
