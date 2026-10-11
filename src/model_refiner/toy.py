"""Offline toy backend. It proves the experiment plumbing end to end, NOT LLM performance."""
from __future__ import annotations

import json
import re

from .backend import ModelBackend
from .types import ModelResult


class ToyBackend(ModelBackend):
    """A deliberately weak frozen model that improves when given rules or relevant demonstrations."""

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        developer = messages[0]["content"]
        user = messages[-1]["content"].lower()

        # Rule compiler behavior.
        if "observed baseline errors" in user:
            return ModelResult(
                "- Treat password, login, locked-out, and sign-in problems as ACCOUNT.\n"
                "- Treat tracking, package, delivery, and shipment problems as SHIPPING.\n"
                "- Treat crash, freeze, error, install, and broken-feature problems as TECH.\n"
                "- Treat charge, refund, invoice, card, and payment problems as BILLING."
            )

        # Learned rules can fix the baseline.
        if "REFINEMENT RULES:" in developer:
            mapping = {
                "ACCOUNT": ["password", "login", "locked", "sign in", "signin"],
                "SHIPPING": ["tracking", "package", "delivery", "shipment", "arrive"],
                "TECH": ["crash", "freeze", "error", "install", "broken", "won't open"],
                "BILLING": ["charge", "refund", "invoice", "card", "payment", "billed"],
            }
            for label, words in mapping.items():
                if any(w in user for w in words):
                    return ModelResult(label)

        # Retrieved demonstrations: use nearest demo label when the query shares a meaningful term.
        demos = []
        block = re.search(r"<refiner_examples_json>\n(.*?)\n</refiner_examples_json>", developer, flags=re.S)
        if block:
            try:
                demos = [(x["input"], x["correct_output"]) for x in json.loads(block.group(1))]
            except (ValueError, KeyError, TypeError):
                demos = []
        stop = {"the", "a", "an", "i", "my", "is", "to", "of", "and", "it", "for", "with", "can", "not"}
        user_tokens = set(re.findall(r"[a-z]+", user)) - stop
        best = (0, None)
        for text, label in demos:
            toks = set(re.findall(r"[a-z]+", text.lower())) - stop
            overlap = len(user_tokens & toks)
            if overlap > best[0]:
                best = (overlap, label.strip())
        if best[1]:
            return ModelResult(best[1])

        # Deliberately weak baseline.
        if "refund" in user or "invoice" in user:
            return ModelResult("BILLING")
        if "password" in user:
            return ModelResult("ACCOUNT")
        return ModelResult("TECH")
