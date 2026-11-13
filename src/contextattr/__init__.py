"""ContextAttr: which parts of a prompt make a model give an output (docs/interface.md, docs/methods.md)."""

from .cache import Cache
from . import responses
from .demo import demo
from .evaluate import faithfulness, intervene
from .methods import LIME, Banzhaf, ContextCite, KernelSHAP, LeaveOneOut
from .prompt import Group, Message, Prompt, Response, Unit, read_prompts, split, write_prompts
from .responses import Verdict, sample
from .result import Explanation
from .run import Results, Run, attribute, explain
from .targets import Choice, Contrast, LogProb, Pooled, Sampled
from .value import Value
from .wrapper import ModelReport, Sampling, Scored, ToyModel, WrapperError, check_model

__version__ = "0.1.3"

checks = responses   # ca.checks.regex(...), ca.checks.reference(...)

__all__ = ["LIME", "Banzhaf", "Cache", "Choice", "ContextCite", "Contrast", "Explanation", "Group", "KernelSHAP",
           "LeaveOneOut", "LogProb", "Message", "ModelReport", "Pooled", "Prompt", "Response", "Results", "Run", "Sampled", "Sampling",
           "Scored", "ToyModel", "Unit", "Value", "Verdict", "WrapperError", "attribute", "check_model", "checks",
           "demo", "explain", "faithfulness", "intervene", "read_prompts", "responses", "sample", "split", "write_prompts"]
