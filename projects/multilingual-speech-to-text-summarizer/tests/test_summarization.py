from src.summarization import summarize_text


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [{"generated_text": f"Summary: {text[:35]}"}]


def test_empty_input_does_not_call_model():
    model = FakeSummarizer()
    assert summarize_text("   ", model) == ""
    assert model.calls == []


def test_summarizes_nonempty_input():
    model = FakeSummarizer()
    result = summarize_text("A meeting covered the project timeline.", model)
    assert result.startswith("Summary:")
    assert len(model.calls) == 1
