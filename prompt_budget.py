"""Conservative offline bounds for the professor's byte-BPE Llama3 model.

UTF-8 byte length bounds byte-BPE token length from above. This deliberately
avoids an external tokenizer download and reserves room for template/output.
It is a safe upper bound, not a claim to reproduce the model's exact tokenizer.
"""
CONTEXT_TOKENS = 2048
OUTPUT_TOKENS = 256
TEMPLATE_RESERVE = 128
PROMPT_BUDGET = CONTEXT_TOKENS - OUTPUT_TOKENS - TEMPLATE_RESERVE


class PromptBudgetError(ValueError):
    pass


def prompt_cost(text: str) -> int:
    return len(text.encode("utf-8"))


def validate_prompt(prompt: str) -> str:
    if prompt_cost(prompt) > PROMPT_BUDGET:
        raise PromptBudgetError(
            "질문이나 번역할 문장이 너무 깁니다. 내용을 나누어 입력해 주세요."
        )
    return prompt
