"""Responses and labels (docs/interface.md §4): checks that label a response, and replies sampled from the model."""

from .checks import Verdict, describe, normalise, reference, regex, verdict
from .sample import sample

__all__ = ["Verdict", "describe", "normalise", "reference", "regex", "sample", "verdict"]
