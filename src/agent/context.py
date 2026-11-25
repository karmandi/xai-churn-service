from typing import Sequence


def estimate_tokens(text: str) -> int:
    """Deterministic token estimator (~4 characters per token)."""
    return len(text) // 4


class ContextManager:
    """Manages conversation turns within a strict token budget."""

    def __init__(self, max_tokens: int = 1200) -> None:
        self.max_tokens = max_tokens
        self.turns: list[dict[str, str]] = []

    @property
    def total_tokens(self) -> int:
        return sum(estimate_tokens(t["content"]) for t in self.turns)

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self._compact_if_needed()

    def _truncate_to_tokens(self, text: str, max_tok: int) -> str:
        max_chars = max(0, max_tok * 4)
        if len(text) <= max_chars:
            return text
        suffix = "... [TRUNCATED]"
        if max_chars <= len(suffix):
            return text[:max_chars]
        return text[: max_chars - len(suffix)] + suffix

    def _compact_if_needed(self) -> None:
        threshold = int(self.max_tokens * 0.8)
        if self.total_tokens <= threshold:
            return

        if len(self.turns) > 2:
            has_system = self.turns[0]["role"] == "system"
            first_turn = [self.turns[0]] if has_system else []
            middle_start = 1 if has_system else 0
            middle_turns = self.turns[middle_start:-1]
            latest_turn = self.turns[-1]

            if middle_turns:
                summary_parts = [
                    f"{t['role']}: {t['content'][:120]}" for t in middle_turns
                ]
                summary_text = "[COMPACTED SUMMARY] " + " | ".join(summary_parts)

                fixed_tokens = sum(
                    estimate_tokens(t["content"]) for t in [*first_turn, latest_turn]
                )
                available_for_summary = max(16, self.max_tokens - fixed_tokens)
                summary_text = self._truncate_to_tokens(
                    summary_text, available_for_summary
                )
                self.turns = [
                    *first_turn,
                    {"role": "system", "content": summary_text},
                    latest_turn,
                ]

        # Guarantee invariant: total_tokens <= max_tokens even if latest turn is huge.
        if self.total_tokens > self.max_tokens and self.turns:
            other_tokens = sum(
                estimate_tokens(t["content"]) for t in self.turns[:-1]
            )
            allowed_latest = max(0, self.max_tokens - other_tokens)
            self.turns[-1]["content"] = self._truncate_to_tokens(
                self.turns[-1]["content"], allowed_latest
            )

    def get_prompt_context(self) -> list[dict[str, str]]:
        self._compact_if_needed()
        return list(self.turns)
