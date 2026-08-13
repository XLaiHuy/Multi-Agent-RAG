import re


def clean_text(text: str) -> str:
    """
    Cleans text safely:
    - Normalizes line endings \r\n to \n.
    - Collapses 3+ consecutive newlines to 2.
    - Strips trailing spaces on each line.
    - Trims leading and trailing whitespace.
    - Preserves headings, Markdown structures, and punctuation.
    """
    if not text:
        raise ValueError("Text to clean cannot be empty.")

    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse multi-newlines (3 or more -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple inline spaces/tabs (keep single space)
    text = re.sub(r"[ \t]{2,}", " ", text)

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text became empty after cleaning.")

    return cleaned
