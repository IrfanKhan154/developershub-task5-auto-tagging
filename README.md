# Task 5 - Auto Tagging Support Tickets using LLM

This project classifies support tickets into categories using an LLM.

## What it does
- Takes a support ticket from CLI input
- Uses **zero-shot** and **few-shot** prompting
- Returns the **top 3 tags**

Default candidate tags:
- billing
- technical_issue
- account_access
- feature_request
- bug_report
- refund
- shipping
- subscription
- password_reset
- general_inquiry

## Files
- `tagging.py` - main CLI script
- `requirements.txt` - dependencies
- `README.md` - documentation

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```
3. Run (interactive input):
   ```bash
   python tagging.py
   ```

You can also pass ticket text directly:
```bash
python tagging.py --ticket "I was charged twice and want a refund"
```

Optional flags:
- `--mode zero-shot|few-shot|both` (default: `both`)
- `--model gpt-4o-mini` (default: `gpt-4o-mini`)

## Example
Input:
```text
My login fails, and password reset email never arrives.
```

Output:
```text
Top 3 tags: account_access, password_reset, technical_issue
```
