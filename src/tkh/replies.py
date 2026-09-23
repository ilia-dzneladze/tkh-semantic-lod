"""Reading a labeller's or rater's reply, which may come from any LLM or
a person: pure JSON, JSON inside a ``` fence, or JSON after a sentence of
preamble. Returns the parsed object or raises ValueError saying why."""
import json
import re


def extract_json(text, expect):
    """expect: list or dict, the top-level type the reply must have."""
    text = text.strip()
    candidates = [text]
    candidates += re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    opener, closer = ("[", "]") if expect is list else ("{", "}")
    start, end = text.find(opener), text.rfind(closer)
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, expect):
            return obj
    raise ValueError(f"no JSON {expect.__name__} found in the reply")
