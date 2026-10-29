"""The deterministic verifier: each tool's checks live in its own module, on contract instances."""

from contract import Task, Tool

from . import check, describe, exact, find, signatures

TOOLS = {
    Tool.DESCRIBE_CORPUS: describe,
    Tool.BUILD_SIGNATURES: signatures,
    Tool.CHECK_PAIRS: check,
    Tool.EXACT_GAPS: exact,
}
# task -> its output's check, verify_task(arguments, output, reference, workspace)
TASKS = {Task.FIND_CONCEPT_GAPS: find.verify_task}


def verify_input(tool, value):
    return TOOLS[Tool(tool)].verify_input(value)


def verify_output(tool, input_value, output_value, workspace):
    return TOOLS[Tool(tool)].verify_output(input_value, output_value, workspace)


def verify_task(task, arguments, output, reference, workspace):
    return TASKS[Task(task)](arguments, output, reference, workspace)
