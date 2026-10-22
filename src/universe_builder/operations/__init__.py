"""Pipeline operations, one module each, all returning a `Result`.

Each operation does one thing to one atom and reports what happened. None of
them writes to disk — persistence is the orchestrator's decision, made from the
`Result`. That separation is what stops a failed operation overwriting good data,
which is what B8 was.
"""

from universe_builder.operations.append import process_append_operation
from universe_builder.operations.create import create_atom
from universe_builder.operations.expand import process_expand_operation
from universe_builder.operations.fix import process_fix_operation
from universe_builder.operations.types import Outcome, Result
from universe_builder.operations.verify import process_verify_operation

__all__ = [
    "Outcome",
    "Result",
    "create_atom",
    "process_append_operation",
    "process_expand_operation",
    "process_fix_operation",
    "process_verify_operation",
]
