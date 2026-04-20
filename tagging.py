import argparse
import json
import os
import re
from typing import List

from openai import OpenAI


CANDIDATE_TAGS = [
    "billing",
    "technical_issue",
    "account_access",
    "feature_request",
    "bug_report",
    "refund",
    "shipping",
    "subscription",
    "password_reset",
    "general_inquiry",
]


def build_zero_shot_prompt(ticket: str) -> str:
    return (
        "You are a support ticket classifier.\\n"
        "Task: Read the ticket and select the top 3 most relevant tags from the allowed list.\\n"
        f"Allowed tags: {', '.join(CANDIDATE_TAGS)}\\n"
        "Return ONLY a JSON array with exactly 3 tags in ranked order.\\n"
        f"Ticket: {ticket}"
    )


def build_few_shot_prompt(ticket: str) -> str:
    return (
        "You are a support ticket classifier.\\n"
        "Select top 3 tags from allowed tags and return ONLY a JSON array.\\n"
        f"Allowed tags: {', '.join(CANDIDATE_TAGS)}\\n\\n"
        "Example 1\\n"
        "Ticket: I was charged twice this month and need a refund.\\n"
        'Output: ["billing", "refund", "subscription"]\\n\\n'
        "Example 2\\n"
        "Ticket: The app crashes when I upload a profile image.\\n"
        'Output: ["technical_issue", "bug_report", "account_access"]\\n\\n'
        "Example 3\\n"
        "Ticket: I cannot log in and the reset email never arrives.\\n"
        'Output: ["account_access", "password_reset", "technical_issue"]\\n\\n'
        f"Now classify this ticket:\\nTicket: {ticket}"
    )


def parse_tags(text: str) -> List[str]:
    """Extract up to 3 valid tags from model output."""
    text = text.strip()
    tags: List[str] = []

    # Try direct JSON array first.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            tags = [str(x).strip() for x in data]
    except json.JSONDecodeError:
        # Fallback: grab first JSON-like array from free text.
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    tags = [str(x).strip() for x in data]
            except json.JSONDecodeError:
                pass

    # Keep only allowed tags and deduplicate while preserving order.
    clean: List[str] = []
    seen = set()
    for tag in tags:
        if tag in CANDIDATE_TAGS and tag not in seen:
            clean.append(tag)
            seen.add(tag)
        if len(clean) == 3:
            break

    return clean


def call_llm(client: OpenAI, prompt: str, model: str) -> List[str]:
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    text = response.output_text or ""
    return parse_tags(text)


def classify_ticket(ticket: str, model: str, mode: str) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    if mode == "zero-shot":
        result = call_llm(client, build_zero_shot_prompt(ticket), model)
    elif mode == "few-shot":
        result = call_llm(client, build_few_shot_prompt(ticket), model)
    else:
        zero_shot = call_llm(client, build_zero_shot_prompt(ticket), model)
        few_shot = call_llm(client, build_few_shot_prompt(ticket), model)
        # Merge while preserving priority from few-shot first, then zero-shot.
        result = []
        for tag in few_shot + zero_shot:
            if tag not in result:
                result.append(tag)
            if len(result) == 3:
                break

    # Fallback to general_inquiry if model output is unusable.
    if not result:
        return ["general_inquiry", "technical_issue", "account_access"]

    # Ensure exactly top 3 if possible.
    for tag in CANDIDATE_TAGS:
        if len(result) >= 3:
            break
        if tag not in result:
            result.append(tag)

    return result[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tag support tickets using LLM prompting.")
    parser.add_argument(
        "--ticket",
        type=str,
        help="Support ticket text. If omitted, interactive input is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero-shot", "few-shot", "both"],
        default="both",
        help="Prompting mode: zero-shot, few-shot, or both (default).",
    )
    args = parser.parse_args()

    ticket = args.ticket.strip() if args.ticket else ""
    if not ticket:
        ticket = input("Enter support ticket: ").strip()

    if not ticket:
        raise SystemExit("Ticket cannot be empty.")

    tags = classify_ticket(ticket=ticket, model=args.model, mode=args.mode)
    print("Top 3 tags:", ", ".join(tags))


if __name__ == "__main__":
    main()
