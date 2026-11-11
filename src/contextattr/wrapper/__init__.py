"""What a user's wrapper must do, the check that it does, and a toy wrapper (docs/wrapper.md)."""

from .check import ModelReport, check_model
from .contract import (Generator, Rendered, Sampling, Scored, Scorer, WrapperError, call_generate, call_score,
                       can_generate, can_score, model_name)
from .toy import ToyModel

__all__ = ["Generator", "ModelReport", "Rendered", "Sampling", "Scored", "Scorer", "ToyModel", "WrapperError",
           "call_generate", "call_score", "can_generate", "can_score", "check_model", "model_name"]
