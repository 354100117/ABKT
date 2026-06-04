#!/usr/bin/env python3
"""Generate long test prompts for ABKT benchmarking.

Usage:
    # From Project Gutenberg (auto-download)
    python3 gen_long_prompt.py --source gutenberg --tokens 2048

    # From a local text file
    python3 gen_long_prompt.py --source file --path /path/to/text.txt --tokens 2048

    # Generate a synthetic "needle in a haystack" test
    python3 gen_long_prompt.py --source needle --tokens 2048

    # Just print the prompt
    python3 gen_long_prompt.py --source gutenberg --tokens 2048 --print

    # Save to file
    python3 gen_long_prompt.py --source gutenberg --tokens 2048 --output prompt_2k.txt
"""

import argparse
import sys
import urllib.request


# Approximate chars per token for English text (GPT-style tokenizer)
CHARS_PER_TOKEN = 4

GUTENBERG_BOOKS = {
    "pride":       1342,   # Pride and Prejudice
    "frankenstein": 84,    # Frankenstein
    "alice":        11,    # Alice in Wonderland
    "metamorphosis":5200,  # The Metamorphosis (Kafka, English)
    "art_of_war":   132,   # The Art of War
    "republic":     1497,  # The Republic (Plato)
    "origin":       2009,  # On the Origin of Species
    "war_and_peace":2600,  # War and Peace
}


def download_gutenberg(book_id: int) -> str:
    """Download a book from Project Gutenberg as plain text."""
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        return resp.read().decode("utf-8", errors="replace")
    except Exception:
        # Try alternate URL format
        url2 = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        resp = urllib.request.urlopen(url2, timeout=30)
        return resp.read().decode("utf-8", errors="replace")


def strip_gutenberg_header(text: str) -> str:
    """Remove Gutenberg project header/footer."""
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if "*** START OF" in line or "*** START OF THE PROJECT" in line:
            start = i + 1
            break
    end = len(lines)
    for i, line in enumerate(lines):
        if "*** END OF" in line or "*** END OF THE PROJECT" in line:
            end = i
            break
    return "\n".join(lines[start:end])


def extract_text(text: str, target_tokens: int) -> str:
    """Extract approximately target_tokens worth of text."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    # Clean up whitespace
    text = " ".join(text.split())
    if len(text) > target_chars:
        # Cut at a sentence boundary near the target
        cut = text[:target_chars]
        last_period = cut.rfind(".")
        if last_period > target_chars * 0.8:
            cut = cut[:last_period + 1]
        return cut
    return text


def make_prompt(text: str, task: str = "qa") -> str:
    """Wrap text into a task prompt."""
    if task == "qa":
        return f"""Read the following text carefully, then answer the question at the end.

--- BEGIN TEXT ---
{text}
--- END TEXT ---

Question: What are the main themes and key points discussed in this text? Provide a detailed summary."""
    elif task == "summarize":
        return f"""Please provide a comprehensive summary of the following text.

--- BEGIN TEXT ---
{text}
--- END TEXT ---

Summary:"""
    elif task == "continue":
        return text
    else:
        return text


def generate_needle_prompt(target_tokens: int) -> str:
    """Generate a needle-in-a-haystack test prompt."""
    filler = "The history of human civilization spans thousands of years, marked by remarkable achievements in science, art, and philosophy. From the ancient pyramids of Egypt to the modern skyscrapers of today, humanity has continuously pushed the boundaries of what is possible. The Renaissance period saw an explosion of creativity and intellectual inquiry, while the Industrial Revolution transformed the way people lived and worked. Throughout these periods of change, one constant has remained: the human desire to understand the world and our place in it. "
    needle = "The secret code for the ABKT system is: BLUE-FALCON-7749. "
    question = "What is the secret code mentioned in the text above?"

    # Estimate tokens per repetition
    tokens_per_rep = len(filler) // CHARS_PER_TOKEN
    reps = max(1, target_tokens // tokens_per_rep)

    # Insert needle at ~70% through the text (classic NIAH position)
    needle_pos = int(reps * 0.7)

    parts = []
    for i in range(reps):
        if i == needle_pos:
            parts.append(needle)
        parts.append(filler)

    haystack = "".join(parts)
    return f"""Read the entire text below carefully, then answer the question.

--- BEGIN TEXT ---
{haystack}
--- END TEXT ---

Question: {question}"""


def main():
    parser = argparse.ArgumentParser(description="Generate long test prompts")
    parser.add_argument("--source", choices=["gutenberg", "file", "needle"],
                        default="needle", help="Text source")
    parser.add_argument("--book", default="pride",
                        choices=list(GUTENBERG_BOOKS.keys()),
                        help="Gutenberg book to use")
    parser.add_argument("--path", help="Path to local text file")
    parser.add_argument("--tokens", type=int, default=2048,
                        help="Target token count")
    parser.add_argument("--task", choices=["qa", "summarize", "continue"],
                        default="qa", help="Task type")
    parser.add_argument("--output", help="Save prompt to file")
    parser.add_argument("--print", action="store_true",
                        help="Print prompt to stdout")
    args = parser.parse_args()

    if args.source == "gutenberg":
        book_id = GUTENBERG_BOOKS[args.book]
        print(f"Downloading '{args.book}' from Project Gutenberg (id={book_id})...",
              file=sys.stderr)
        raw = download_gutenberg(book_id)
        text = strip_gutenberg_header(raw)
        text = extract_text(text, args.tokens)
        prompt = make_prompt(text, args.task)
    elif args.source == "file":
        if not args.path:
            print("Error: --path required with --source file", file=sys.stderr)
            sys.exit(1)
        with open(args.path) as f:
            text = f.read()
        text = extract_text(text, args.tokens)
        prompt = make_prompt(text, args.task)
    elif args.source == "needle":
        prompt = generate_needle_prompt(args.tokens)

    est_tokens = len(prompt) // CHARS_PER_TOKEN
    print(f"Generated prompt: ~{est_tokens} tokens ({len(prompt)} chars)",
          file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            f.write(prompt)
        print(f"Saved to {args.output}", file=sys.stderr)

    if args.print or not args.output:
        print(prompt)


if __name__ == "__main__":
    main()
