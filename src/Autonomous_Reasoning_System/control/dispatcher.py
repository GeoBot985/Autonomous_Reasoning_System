
import time
import logging
import traceback
from typing import Any, Dict, Callable, List, Optional, Union

logger = logging.getLogger(__name__)

class Dispatcher:
    """
    Central dispatcher for tool execution.

    Responsibilities:
    - Resolve tool by name
    - Validate inputs (schema / type / missing args)
    - Enforce execution pattern
    - Attach context metadata
    - Return standardised {status, data, errors, warnings, meta}
    - Track lineage and run metadata
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []

    def register_tool(self, name: str, handler: Callable, schema: Optional[Dict[str, Any]] = None):
        """
        Registers a tool with a name, handler, and optional schema.

        Args:
            name: The unique name of the tool.
            handler: The callable function or method to execute.
            schema: Optional dictionary defining expected arguments.
                    Format: {"arg_name": {"type": type_class, "required": bool}}
        """
        if name in self._tools:
            logger.warning(f"Overwriting tool '{name}'")

        self._tools[name] = {
            "handler": handler,
            "schema": schema or {}
        }
        logger.info(f"Registered tool: {name}")

    def dispatch(self, tool_name: str, arguments: Dict[str, Any] = None, dry_run: bool = False, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes a tool by name.

        Args:
            tool_name: The name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.
            dry_run: If True, validates inputs but does not execute the tool.
            context: Optional context metadata to attach to the run.

        Returns:
            Standardized response dictionary:
            {
                "status": "success" | "error",
                "data": Any,
                "errors": List[str],
                "warnings": List[str],
                "meta": Dict[str, Any]
            }
        """
        arguments = arguments or {}
        context = context or {}

        start_time = time.time()
        warnings: List[str] = []
        errors: List[str] = []
        status = "success"
        output: Any = None

        # 1. Resolve tool
        tool_def = self._tools.get(tool_name)
        if not tool_def:
            error_msg = f"Tool '{tool_name}' not found"
            errors.append(error_msg)
            return self._finalize_response(
                tool_name=tool_name,
                status="error",
                output=None,
                errors=errors,
                warnings=warnings,
                context=context,
                arguments=arguments,
                start_time=start_time
            )

        # 2. Validate inputs
        validation_errors = self._validate_inputs(tool_def["schema"], arguments)
        if validation_errors:
            errors.extend(validation_errors)
            return self._finalize_response(
                tool_name=tool_name,
                status="error",
                output=None,
                errors=errors,
                warnings=warnings,
                context=context,
                arguments=arguments,
                start_time=start_time
            )

        # 3. Execution
        if dry_run:
            output = "Dry run successful. Tool would execute with provided arguments."
        else:
            try:
                handler = tool_def["handler"]
                # We pass arguments as kwargs.
                output = handler(**arguments)
            except Exception as e:
                status = "error"
                errors.append(str(e))
                # Include traceback in metadata?
                # For now, keeping it simple as requested.
                logger.error(f"Error executing tool '{tool_name}': {e}")
                logger.debug(traceback.format_exc())

        # 4. Finalize and return response (includes logging/lineage)
        return self._finalize_response(
            tool_name=tool_name,
            status=status,
            output=output,
            errors=errors,
            warnings=warnings,
            context=context,
            arguments=arguments,
            start_time=start_time
        )

    def _validate_inputs(self, schema: Dict[str, Any], arguments: Dict[str, Any]) -> List[str]:
        errors = []
        for arg_name, rules in schema.items():
            # Check for required arguments
            if rules.get("required", False) and arg_name not in arguments:
                errors.append(f"Missing required argument: {arg_name}")
                continue

            # Check type if argument is present
            if arg_name in arguments:
                val = arguments[arg_name]
                expected_type = rules.get("type")

                # Handle typing.Union or similar complex types if possible, but sticking to basic types for now
                # or allow user to pass a tuple of types as expected_type (standard isinstance behavior)
                if expected_type:
                    try:
                        if not isinstance(val, expected_type):
                            errors.append(f"Argument '{arg_name}' expected type {expected_type}, got {type(val).__name__}")
                    except TypeError:
                         # In case expected_type is not a class/tuple/type, we skip strict check or warn
                         # But let's assume the user registers with valid types.
                         pass

        return errors

    def _finalize_response(self, tool_name: str, status: str, output: Any, errors: List[str], warnings: List[str], context: Dict[str, Any], arguments: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        duration = time.time() - start_time

        meta = {
            "tool_name": tool_name,
            "timestamp": start_time,
            "duration": duration,
            "context": context
        }

        # Lineage / Run Metadata
        record = {
            "timestamp": start_time,
            "tool_name": tool_name,
            "input_summary": str(arguments),
            "output_summary": str(output)[:200] if output is not None else "None",
            "warnings": list(warnings), # copy
            "errors": list(errors), # copy
            "duration": duration,
            "status": status
        }
        self._history.append(record)

        return {
            "status": status,
            "data": output,
            "errors": errors,
            "warnings": warnings,
            "meta": meta
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the full history of tool executions."""
        return self._history
