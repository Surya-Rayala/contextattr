"""The input: prompts, units, splitting, masking and prompts.jsonl (docs/interface.md §2)."""

from .io import prompt_from_dict, prompt_to_dict, read_prompts, write_prompts
from .render import as_mask, render
from .split import KINDS, split
from .units import LABELS, Group, Message, Piece, Prompt, Response, Unit

__all__ = ["KINDS", "LABELS", "Group", "Message", "Piece", "Prompt", "Response", "Unit", "as_mask",
           "prompt_from_dict", "prompt_to_dict", "read_prompts", "render", "split", "write_prompts"]
