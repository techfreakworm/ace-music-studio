"""``app._safe_call`` turns handler exceptions into ``gr.Error`` with a useful message.

On ZeroGPU an exception that is not a ``gr.Error`` reaches the browser as a
bare class name, so every failure must be mapped.
"""

from __future__ import annotations

import os

import pytest

if os.getenv("SPACE_ID"):  # importing app would run the real bootstrap
    pytest.skip("SPACE_ID is set in this environment", allow_module_level=True)

import gradio as gr

import app
import lora_stack


def _raise(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _message(err: gr.Error) -> str:
    return err.message


def test_safe_call_returns_handler_result():
    assert app._safe_call(lambda a, b=0: a + b, 2, b=3) == 5


def test_gr_error_passes_through_unchanged():
    original = gr.Error("Quota exceeded", duration=7)
    with pytest.raises(gr.Error) as exc:
        app._safe_call(_raise(original))
    assert exc.value is original
    assert _message(exc.value) == "Quota exceeded"


def test_known_exceptions_keep_their_messages():
    cases = [
        (lora_stack.LoRAValidationError("bad rank"), "bad rank"),
        (ValueError("prompt is required"), "prompt is required"),
        (FileNotFoundError("ref.wav"), "File not found: ref.wav"),
        (RuntimeError("boom"), "Generation failed: boom"),
    ]
    for err, expected in cases:
        with pytest.raises(gr.Error) as exc:
            app._safe_call(_raise(err))
        assert _message(exc.value) == expected
        assert exc.value.__cause__ is err


def test_mps_runtime_error_keeps_its_hint():
    with pytest.raises(gr.Error) as exc:
        app._safe_call(_raise(RuntimeError("op not implemented for MPS")))
    assert _message(exc.value).startswith("Apple Silicon op issue: op not implemented for MPS")


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (
            OSError("libtorchaudio.so: undefined symbol: torch_exception_get_what"),
            "OSError: libtorchaudio.so",
        ),
        (KeyError("vocal_language"), "KeyError: 'vocal_language'"),
        (ImportError("No module named 'torchcodec'"), "ImportError: No module named 'torchcodec'"),
        (TypeError("unexpected keyword 'foo'"), "TypeError: unexpected keyword 'foo'"),
    ],
)
def test_unmapped_exceptions_become_gr_error_with_type_and_message(err, expected):
    with pytest.raises(gr.Error) as exc:
        app._safe_call(_raise(err))
    assert _message(exc.value).startswith(expected)
    assert exc.value.__cause__ is err


def test_base_exceptions_are_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        app._safe_call(_raise(KeyboardInterrupt()))
