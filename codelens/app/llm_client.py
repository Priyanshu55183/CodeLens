"""
LLM client with configurable provider (Phase 5: multi-LLM).

Design decisions:
- Single function `ask_llm()` hides the provider-specific API behind a
  uniform interface. Downstream code never imports openai/anthropic/gemini
  directly — it always goes through this module.
- Lazy imports: provider libraries (anthropic, google-generativeai) are
  imported inside their respective _ask_* functions, not at module level.
  This means if a user only has openai installed, the app starts fine and
  only fails when they actually try to use anthropic/gemini. Clean UX.
- temperature=0 everywhere by default. We want deterministic, grounded answers.
  Some providers cap temperature at 1.0 (Gemini), we clamp accordingly.
- The system prompt is passed in by the caller (not hardcoded here).
  This module is just the transport layer — Strategy pattern.
- Anthropic uses a separate `system` parameter (not a message role).
- Gemini uses google.generativeai with GenerationConfig for temperature/tokens.
  System instruction is passed as `system_instruction` to GenerativeModel.

Interview note: This is the "Strategy pattern" — swapping implementations
without changing the calling code. The env var (LLM_PROVIDER) picks the strategy.

Supported providers:
  openai    → gpt-4o-mini (default), gpt-4o, gpt-3.5-turbo, etc.
  anthropic → claude-3-5-haiku-20241022 (default), claude-3-5-sonnet-20241022, etc.
  gemini    → gemini-1.5-flash (default), gemini-1.5-pro, etc.
"""

import logging

logger = logging.getLogger(__name__)

# Default models per provider — used in .env.example and CLI help text
PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini": "gemini-1.5-flash",
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_DEFAULTS.keys())


def ask_llm(
    prompt: str,
    system_prompt: str = "",
    provider: str = "openai",
    model: str = "",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Send a prompt to the configured LLM and return the response text.

    Args:
        prompt: The user/main prompt.
        system_prompt: Optional system-level instructions for the LLM.
        provider: LLM provider ("openai", "anthropic", "gemini").
        model: Model name. If empty, uses the provider's default from PROVIDER_DEFAULTS.
        api_key: API key for the provider.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        The LLM's response text.

    Raises:
        ValueError: If the provider is not supported.
        ImportError: If the provider's library is not installed.
    """
    resolved_model = model or PROVIDER_DEFAULTS.get(provider, "")

    logger.info("ask_llm: provider=%s model=%s", provider, resolved_model)

    if provider == "openai":
        return _ask_openai(prompt, system_prompt, resolved_model, api_key, temperature, max_tokens)
    elif provider == "anthropic":
        return _ask_anthropic(prompt, system_prompt, resolved_model, api_key, temperature, max_tokens)
    elif provider == "gemini":
        return _ask_gemini(prompt, system_prompt, resolved_model, api_key, temperature, max_tokens)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}. "
            f"Set LLM_PROVIDER in your .env file."
        )


# ── OpenAI ───────────────────────────────────────────────────────────────────

def _ask_openai(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call OpenAI's chat completion API."""
    from openai import OpenAI  # lazy import (already a hard dep, but consistent style)

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


# ── Anthropic ────────────────────────────────────────────────────────────────

def _ask_anthropic(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Anthropic's Messages API (Claude).

    Anthropic's API differs from OpenAI in two key ways:
    1. The system prompt is a top-level parameter, not a message role.
    2. max_tokens is required (not optional).
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The 'anthropic' package is required to use the Anthropic provider. "
            "Install it with: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    # Anthropic returns a list of content blocks; we join all text blocks.
    text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    logger.debug("Anthropic response (%d chars): %s...", len(text), text[:100])
    return text


# ── Gemini ───────────────────────────────────────────────────────────────────

def _ask_gemini(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Google's Gemini API.

    Gemini differs from OpenAI:
    1. System instruction is set at model-init time, not per-message.
    2. Uses GenerationConfig for temperature and max_output_tokens.
    3. Temperature must be in [0.0, 2.0]; we clamp to be safe.
    4. The SDK uses generate_content() not chat.completions.create().
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "The 'google-generativeai' package is required to use the Gemini provider. "
            "Install it with: pip install google-generativeai"
        ) from e

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=min(max(temperature, 0.0), 2.0),  # Gemini clamps at 2.0
        max_output_tokens=max_tokens,
    )

    model_kwargs: dict = {"model_name": model, "generation_config": generation_config}
    if system_prompt:
        model_kwargs["system_instruction"] = system_prompt

    gemini_model = genai.GenerativeModel(**model_kwargs)
    response = gemini_model.generate_content(prompt)

    text = response.text or ""
    logger.debug("Gemini response (%d chars): %s...", len(text), text[:100])
    return text

