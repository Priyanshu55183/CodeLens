"""
Tests for the multi-LLM client (Phase 5).

All tests mock the provider SDK calls — no real API calls are made.
Tests cover:
- OpenAI: happy path, system prompt, default model fallback
- Anthropic: happy path, system prompt included/excluded, content block joining
- Gemini: happy path, system_instruction, temperature clamping
- ask_llm routing: correct provider function called, model defaulting
- Error cases: unsupported provider, missing library (ImportError)
- Settings: active_api_key, resolved_llm_model properties
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from codelens.app.llm_client import (
    ask_llm,
    PROVIDER_DEFAULTS,
    SUPPORTED_PROVIDERS,
    _ask_openai,
    _ask_anthropic,
    _ask_gemini,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_openai_response(text: str) -> MagicMock:
    """Build a minimal mock of an OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_anthropic_response(text: str) -> MagicMock:
    """Build a minimal mock of an Anthropic Messages response."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_gemini_response(text: str) -> MagicMock:
    """Build a minimal mock of a Gemini generate_content response."""
    response = MagicMock()
    response.text = text
    return response


# ── PROVIDER_DEFAULTS & constants ─────────────────────────────────────────────

class TestProviderDefaults:
    def test_all_three_providers_have_defaults(self):
        assert "openai" in PROVIDER_DEFAULTS
        assert "anthropic" in PROVIDER_DEFAULTS
        assert "gemini" in PROVIDER_DEFAULTS

    def test_supported_providers_tuple(self):
        assert set(SUPPORTED_PROVIDERS) == {"openai", "anthropic", "gemini"}

    def test_openai_default_model(self):
        assert PROVIDER_DEFAULTS["openai"] == "gpt-4o-mini"

    def test_anthropic_default_model(self):
        assert "claude" in PROVIDER_DEFAULTS["anthropic"]

    def test_gemini_default_model(self):
        assert "gemini" in PROVIDER_DEFAULTS["gemini"]


# ── ask_llm routing ───────────────────────────────────────────────────────────

class TestAskLLMRouting:
    """Test that ask_llm routes to the right backend and handles model defaults."""

    @patch("codelens.app.llm_client._ask_openai")
    def test_routes_to_openai(self, mock_openai):
        mock_openai.return_value = "openai answer"
        result = ask_llm("hello", provider="openai", api_key="sk-test")
        assert result == "openai answer"
        mock_openai.assert_called_once()

    @patch("codelens.app.llm_client._ask_anthropic")
    def test_routes_to_anthropic(self, mock_anthropic):
        mock_anthropic.return_value = "claude answer"
        result = ask_llm("hello", provider="anthropic", api_key="ant-test")
        assert result == "claude answer"
        mock_anthropic.assert_called_once()

    @patch("codelens.app.llm_client._ask_gemini")
    def test_routes_to_gemini(self, mock_gemini):
        mock_gemini.return_value = "gemini answer"
        result = ask_llm("hello", provider="gemini", api_key="gem-test")
        assert result == "gemini answer"
        mock_gemini.assert_called_once()

    @patch("codelens.app.llm_client._ask_openai")
    def test_empty_model_uses_provider_default(self, mock_openai):
        mock_openai.return_value = "ok"
        ask_llm("hello", provider="openai", model="", api_key="sk-test")
        # Second arg to _ask_openai is system_prompt, third is model
        called_model = mock_openai.call_args[0][2]
        assert called_model == PROVIDER_DEFAULTS["openai"]

    @patch("codelens.app.llm_client._ask_anthropic")
    def test_explicit_model_overrides_default(self, mock_anthropic):
        mock_anthropic.return_value = "ok"
        ask_llm("hello", provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="ant")
        called_model = mock_anthropic.call_args[0][2]
        assert called_model == "claude-3-5-sonnet-20241022"

    def test_unsupported_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            ask_llm("hello", provider="cohere")

    def test_unsupported_provider_error_lists_valid_providers(self):
        with pytest.raises(ValueError, match="openai"):
            ask_llm("hello", provider="unknown_provider")


# ── OpenAI backend ────────────────────────────────────────────────────────────

class TestAskOpenAI:
    @patch("openai.OpenAI")
    def test_returns_response_text(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("Hello!")

        result = _ask_openai("Tell me about Python", "", "gpt-4o-mini", "sk-key", 0.0, 1024)
        assert result == "Hello!"

    @patch("openai.OpenAI")
    def test_system_prompt_included_as_system_message(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("ok")

        _ask_openai("question", "You are CodeLens.", "gpt-4o-mini", "sk", 0.0, 512)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are CodeLens."}
        assert messages[1]["role"] == "user"

    @patch("openai.OpenAI")
    def test_no_system_prompt_sends_only_user_message(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("ok")

        _ask_openai("question", "", "gpt-4o-mini", "sk", 0.0, 512)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("openai.OpenAI")
    def test_null_content_returns_empty_string(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        resp = _make_openai_response(None)
        mock_client.chat.completions.create.return_value = resp

        result = _ask_openai("q", "", "gpt-4o-mini", "sk", 0.0, 512)
        assert result == ""


# ── Anthropic backend ─────────────────────────────────────────────────────────

class TestAskAnthropic:
    def _mock_anthropic_module(self) -> MagicMock:
        """Build a minimal mock of the anthropic library."""
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.Anthropic.return_value = mock_client
        return mock_module, mock_client

    def test_returns_response_text(self):
        mock_module, mock_client = self._mock_anthropic_module()
        mock_client.messages.create.return_value = _make_anthropic_response("Claude says hi")

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            result = _ask_anthropic("hello", "", "claude-3-5-haiku-20241022", "ant-key", 0.0, 1024)
        assert result == "Claude says hi"

    def test_system_prompt_passed_as_top_level_param(self):
        mock_module, mock_client = self._mock_anthropic_module()
        mock_client.messages.create.return_value = _make_anthropic_response("ok")

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            _ask_anthropic("question", "You are CodeLens.", "claude-3-5-haiku-20241022", "ant", 0.0, 512)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are CodeLens."
        # system should NOT appear in messages list
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    def test_no_system_prompt_omits_system_key(self):
        mock_module, mock_client = self._mock_anthropic_module()
        mock_client.messages.create.return_value = _make_anthropic_response("ok")

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            _ask_anthropic("question", "", "claude-3-5-haiku-20241022", "ant", 0.0, 512)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_multiple_content_blocks_joined(self):
        mock_module, mock_client = self._mock_anthropic_module()
        block1 = MagicMock()
        block1.text = "Part 1. "
        block2 = MagicMock()
        block2.text = "Part 2."
        response = MagicMock()
        response.content = [block1, block2]
        mock_client.messages.create.return_value = response

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            result = _ask_anthropic("q", "", "claude-3-5-haiku-20241022", "ant", 0.0, 512)
        assert result == "Part 1. Part 2."

    def test_missing_library_raises_import_error(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                _ask_anthropic("q", "", "claude-3-5-haiku-20241022", "ant", 0.0, 512)


# ── Gemini backend ────────────────────────────────────────────────────────────

class TestAskGemini:
    def _mock_genai_module(self) -> tuple[MagicMock, MagicMock]:
        mock_module = MagicMock()
        mock_model_instance = MagicMock()
        mock_module.GenerativeModel.return_value = mock_model_instance
        mock_module.GenerationConfig = MagicMock(return_value=MagicMock())
        return mock_module, mock_model_instance

    def _patch_genai(self, mock_module: MagicMock) -> dict:
        """Build sys.modules patch dict so 'import google.generativeai as genai' resolves to mock_module."""
        mock_google = MagicMock()
        mock_google.generativeai = mock_module
        return {"google": mock_google, "google.generativeai": mock_module}

    def test_returns_response_text(self):
        mock_module, mock_model_instance = self._mock_genai_module()
        mock_model_instance.generate_content.return_value = _make_gemini_response("Gemini says hi")

        with patch.dict("sys.modules", self._patch_genai(mock_module)):
            result = _ask_gemini("hello", "", "gemini-1.5-flash", "gem-key", 0.0, 1024)
        assert result == "Gemini says hi"

    def test_system_instruction_passed_when_system_prompt_given(self):
        mock_module, mock_model_instance = self._mock_genai_module()
        mock_model_instance.generate_content.return_value = _make_gemini_response("ok")

        with patch.dict("sys.modules", self._patch_genai(mock_module)):
            _ask_gemini("question", "You are CodeLens.", "gemini-1.5-flash", "gem", 0.0, 512)
            call_kwargs = mock_module.GenerativeModel.call_args[1]
            assert call_kwargs.get("system_instruction") == "You are CodeLens."

    def test_no_system_prompt_omits_system_instruction(self):
        mock_module, mock_model_instance = self._mock_genai_module()
        mock_model_instance.generate_content.return_value = _make_gemini_response("ok")

        with patch.dict("sys.modules", self._patch_genai(mock_module)):
            _ask_gemini("question", "", "gemini-1.5-flash", "gem", 0.0, 512)
            call_kwargs = mock_module.GenerativeModel.call_args[1]
            assert "system_instruction" not in call_kwargs

    def test_temperature_clamped_to_valid_range(self):
        mock_module, mock_model_instance = self._mock_genai_module()
        mock_model_instance.generate_content.return_value = _make_gemini_response("ok")

        with patch.dict("sys.modules", self._patch_genai(mock_module)):
            # Temperature -1.0 should be clamped to 0.0
            _ask_gemini("q", "", "gemini-1.5-flash", "gem", -1.0, 512)
            gen_config_call = mock_module.GenerationConfig.call_args[1]
            assert gen_config_call["temperature"] == 0.0

    def test_missing_library_raises_import_error(self):
        with patch.dict("sys.modules", {"google": None, "google.generativeai": None}):
            with pytest.raises(ImportError, match="google-generativeai"):
                _ask_gemini("q", "", "gemini-1.5-flash", "gem", 0.0, 512)


# ── Settings.active_api_key & resolved_llm_model ─────────────────────────────

class TestSettingsProperties:
    """Test the new properties on Settings added in Phase 5."""

    def _make_settings(self, **overrides):
        """Create a Settings-like object with overridable fields."""
        from codelens.app.config import Settings
        import os
        env = {
            "OPENAI_API_KEY": "sk-open",
            "ANTHROPIC_API_KEY": "ant-key",
            "GEMINI_API_KEY": "gem-key",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "",
            **{k.upper(): str(v) for k, v in overrides.items()},
        }
        with patch.dict(os.environ, env, clear=False):
            return Settings(_env_file=None)

    def test_active_api_key_openai(self):
        s = self._make_settings(llm_provider="openai")
        assert s.active_api_key == "sk-open"

    def test_active_api_key_anthropic(self):
        s = self._make_settings(llm_provider="anthropic")
        assert s.active_api_key == "ant-key"

    def test_active_api_key_gemini(self):
        s = self._make_settings(llm_provider="gemini")
        assert s.active_api_key == "gem-key"

    def test_resolved_llm_model_uses_explicit_model(self):
        s = self._make_settings(llm_provider="openai", llm_model="gpt-4o")
        assert s.resolved_llm_model == "gpt-4o"

    def test_resolved_llm_model_falls_back_to_default(self):
        s = self._make_settings(llm_provider="anthropic", llm_model="")
        assert s.resolved_llm_model == PROVIDER_DEFAULTS["anthropic"]
