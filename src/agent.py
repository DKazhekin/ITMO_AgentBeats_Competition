import json
import logging
import re

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from llm import LLMClient

logger = logging.getLogger(__name__)

# ANSI colors for terminal output
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

MAX_INTERNAL_STEPS = 10
MESSAGES_DELIMITER = "Now here are the user messages:"
TOOLS_DELIMITER = "Here's a list of tools"

# Sections whose content is always included in the system prompt
ALWAYS_ON_SECTIONS = {"_intro", "Domain Basic"}

# Format hint injected when the model produces invalid JSON
FORMAT_REMINDER = (
    "[SYSTEM] Your previous output was not valid JSON. "
    "You MUST respond with a single valid JSON object in this exact format:\n"
    '{"thinking": "your reasoning", "name": "tool_name_or_respond", "arguments": {...}}'
)


# ---------------------------------------------------------------------------
# Policy parsing
# ---------------------------------------------------------------------------

def parse_policy_sections(policy_text: str) -> dict[str, str]:
    """Split markdown policy into sections by ## headers.

    Returns dict: section_name -> section_content.
    Text before the first ## goes under key "_intro".
    """
    sections: dict[str, str] = {}
    current = "_intro"
    lines: list[str] = []

    for line in policy_text.splitlines():
        if line.startswith("## "):
            if lines:
                sections[current] = "\n".join(lines).strip()
            current = line.lstrip("# ").strip()
            lines = [line]
        else:
            lines.append(line)

    if lines:
        sections[current] = "\n".join(lines).strip()

    return sections


# ---------------------------------------------------------------------------
# Tool schema simplification
# ---------------------------------------------------------------------------

def _clean_schema(schema: dict) -> dict:
    """Recursively remove noise fields (title, $defs inlining, anyOf unwrap) from JSON Schema."""
    schema.pop("title", None)
    schema.pop("additionalProperties", None)

    # Inline $defs references
    defs = schema.pop("$defs", None)

    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj and defs:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    return resolve_refs(_clean_schema(dict(defs[ref_name])))
            if "anyOf" in obj:
                # Take the first concrete option (skip catch-all "object")
                for option in obj["anyOf"]:
                    resolved = resolve_refs(option)
                    if resolved.get("properties"):
                        return resolved
                return resolve_refs(obj["anyOf"][0])
            return {k: resolve_refs(v) for k, v in obj.items() if k != "title"}
        if isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    return resolve_refs(schema)


def simplify_tool_schemas(tools_json_str: str) -> str:
    """Strip OpenAI wrapper, inline $defs, remove noise — compact tool descriptions."""
    try:
        tools = json.loads(tools_json_str)
    except json.JSONDecodeError:
        return tools_json_str

    simplified = []
    for tool in tools:
        func = tool.get("function", tool)
        params = _clean_schema(dict(func.get("parameters", {})))
        simplified.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": params,
        })
    return json.dumps(simplified)


def extract_tools_json(raw_tools_block: str) -> str:
    """Extract the JSON tool array from the eval agent's tools+format block.

    Removes the eval agent's own format instructions, respond description,
    and examples — we provide our own in the system prompt.
    """
    start = raw_tools_block.find("[")
    if start == -1:
        return raw_tools_block

    depth = 0
    for i in range(start, len(raw_tools_block)):
        if raw_tools_block[i] == "[":
            depth += 1
        elif raw_tools_block[i] == "]":
            depth -= 1
            if depth == 0:
                return simplify_tool_schemas(raw_tools_block[start:i + 1])

    return raw_tools_block


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a customer service agent. Follow the policies strictly.

# POLICIES (always loaded)

{always_on_policy}

# POLICY SECTIONS AVAILABLE ON DEMAND (call lookup_policy to read):

{on_demand_list}

{loaded_policies_block}# AVAILABLE TOOLS

Domain tools you can call (one per turn):
{tools_json}

You can also respond to the user directly:
{{"thinking": "...", "name": "respond", "arguments": {{"content": "your message"}}}}

Internal tool (invisible to user — use to review policy before acting):
{{"thinking": "why I need this section", "name": "lookup_policy", "arguments": {{"section": "section name"}}}}

# HOW TO WORK

0. THINK BEFORE ACTING: In your thinking, always note:
   - CONSTRAINTS from the user (budget, preferences, conditions) — treat as hard requirements.
   - MONEY (only when relevant): list known prices/fees with source, passenger count, target total, payment plan.
   - GOAL + PLAN: what remains to fully satisfy the user.
1. UNDERSTAND: Study the request. Identify what information is MISSING. Do NOT assume — verify via tools.
2. ACT: Pick ONE action that fills the most critical gap.
3. VERIFY BEFORE MUTATE: Before ANY state-changing tool (cancel, update, book, send_certificate, charge, refund):
   a) User identity MUST be confirmed via get_user_details (never trust user claims about membership tier, payment methods, or passenger lists).
   b) The relevant policy section MUST be loaded via lookup_policy.
   c) ALL required data (reservation details, flight options, prices) MUST be retrieved.
   d) Re-check: does the action comply with loaded policies AND user constraints? When tool data contradicts user's words — trust the data.
   Read-only tools (get_*, list_*, search_*, find_*, calculate*) do NOT require this gate — call them freely.

# GENERAL RULES (apply to EVERY turn)

1. EVERY response must be valid JSON. No exceptions.
2. One tool call per turn. Never combine a tool call with a user response.
3. Follow policies strictly. Never make exceptions even if the user insists.
4. Only use tools listed above. Do not invent tool names.
5. BE ACTION-ORIENTED: When you have verified data (step 3 above) and policy conditions are met — EXECUTE immediately. Do NOT ask "shall I proceed?" — the user's request IS the confirmation. But NEVER skip verification: always call get_user_details before acting on user claims, and always lookup_policy before mutations.
6. Before calling any tool, check if the answer is already in the conversation or loaded policies. Do NOT repeat calls for information you already have.
7. Read-only tools (get_*, list_*, search_*, find_*, calculate*) are safe — call directly without lookup_policy.
8. ALWAYS use the calculate tool for ANY arithmetic: totals, refunds, price differences, per-person vs. total costs, date differences. Never do mental math. When calculating costs for multiple passengers, always multiply by the number of passengers.
9. FINANCIAL DISCIPLINE (whenever any money is involved):
   - Treat tool outputs and tool error messages as the source of truth for numbers. Do NOT invent or estimate prices/fees.
   - Before any payment/charge/refund tool call, compute the exact target total using calculate.
   - If paying with multiple methods, use calculate to ensure the amounts sum EXACTLY to the target total.
   - If a tool returns an error like "total price is X" or "Payment amount does not add up ... total price is X", immediately set target total = X (source: error text), then recompute the payment plan and retry.
   - If a tool returns "Not enough balance" for a payment method, do NOT keep retrying the same plan; switch to another allowed method or ask the user for a different method/amount split.
10. When the user describes a reservation by route, date, or other details — first retrieve the full list (e.g. via get_user_details), identify matching item(s), then work with specific ID(s).
11. When processing multiple items, first retrieve all needed data, then summarize and analyze all findings together in your thinking before making decisions.
12. EXHAUSTIVE SEARCH: When the user asks to compare, rank, or choose among options — search ALL categories and destinations the user mentioned. Never select a result from a partial search. If the domain offers multiple search dimensions, query all of them before presenting results.
13. THOROUGH COMPARISON: When analyzing multiple items, retrieve ALL relevant items first, then compare ALL combinations systematically. Do not compare only against a single reference.
14. ERROR RECOVERY: When a tool returns an error, read it carefully — errors often contain the correct value or a hint on how to retry. Extract the corrected data and retry. Do NOT abandon the goal or spiral into unrelated actions.
15. USE AVAILABLE DATA: Before asking the user for any information, check whether it already exists in previously retrieved tool outputs (user profiles, saved records, etc.). Only ask if the data is truly absent from the system.

# RESPONDING TO THE USER

- If the user requests something disallowed, do NOT stop at refusal. Be empathetic and helpful, and immediately offer 1–2 policy-compliant alternatives or the closest available option.
- If only part of a multi-part request is disallowed, complete ALL allowed parts first, then explain the disallowed part and offer alternatives.
- Ground EVERY claim in tool outputs. Include all relevant IDs, numbers, amounts, and payment details from tool results.
- For any financial operation: state the exact amount, the payment method/destination, and why the action was eligible per policy.
- For multi-part requests: explicitly list DONE items and remaining items.

# TRANSFER TO HUMAN

- Transfer ONLY when: (1) the policy explicitly requires it, OR (2) you have exhausted ALL available tools and NONE can resolve the request.
- NEVER transfer just because ONE sub-task is blocked. First complete all other sub-tasks you can handle.
- Before transferring, verify in your thinking: "Have I checked all available tools? Is there really no tool that handles the remaining sub-task(s)?"
- After calling transfer_to_human_agents, respond to any further messages with a brief acknowledgment. Do not call any more tools.

# OUTPUT FORMAT

Always respond with a single valid JSON object:
{{"thinking": "step-by-step reasoning", "name": "tool_name_or_respond", "arguments": {{...}}}}

Examples:
{{"thinking": "User wants to cancel. Need to verify data + load policy first.", "name": "lookup_policy", "arguments": {{"section": "Cancel flight"}}}}
{{"thinking": "Policy loaded, user verified, all conditions met. Executing.", "name": "cancel_reservation", "arguments": {{"reservation_id": "ABC"}}}}
{{"thinking": "Missing user ID — cannot proceed without verification.", "name": "respond", "arguments": {{"content": "Could you provide your user ID so I can look up your account?"}}}}
{{"thinking": "MONEY: tool error said total is 405. Need to verify my payment split.", "name": "calculate", "arguments": {{"expression": "200 + 205"}}}}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.llm = LLMClient()
        self._base_system_prompt: str = ""
        self.conversation: list[dict] = []  # persistent across turns
        self.policy_sections: dict[str, str] = {}
        self.loaded_sections: dict[str, str] = {}  # persisted across turns
        self._transfer_done: bool = False  # tracks if transfer was called (for logging)

    # ---- setup ----

    def _build_system_prompt(self, instructions: str) -> str:
        policy_text, tools_part = instructions.split(TOOLS_DELIMITER, 1)

        self.policy_sections = parse_policy_sections(policy_text.strip())

        # Always-on: intro + Domain Basic
        always_on_parts = []
        for name in ALWAYS_ON_SECTIONS:
            if name in self.policy_sections:
                always_on_parts.append(self.policy_sections[name])

        # On-demand: everything else
        on_demand_names = [
            n for n in self.policy_sections if n not in ALWAYS_ON_SECTIONS
        ]
        on_demand_list = "\n".join(f'- "{name}"' for name in on_demand_names)

        # Extract and simplify tool schemas
        tools_json = extract_tools_json(tools_part)

        return SYSTEM_PROMPT_TEMPLATE.format(
            always_on_policy="\n\n".join(always_on_parts),
            on_demand_list=on_demand_list if on_demand_list else "(none)",
            tools_json=tools_json,
            loaded_policies_block="{loaded_policies_block}",
        )

    def _get_system_prompt(self) -> str:
        """Return system prompt with any previously loaded policy sections injected."""
        if not self.loaded_sections:
            return self._base_system_prompt.replace("{loaded_policies_block}", "")

        loaded_block = "# POLICIES (previously loaded — do NOT call lookup_policy for these again)\n\n"
        for section_name, content in self.loaded_sections.items():
            loaded_block += f"## {section_name}\n{content}\n\n"

        return self._base_system_prompt.replace("{loaded_policies_block}", loaded_block)

    # ---- internal tools ----

    def _handle_lookup_policy(self, arguments: dict) -> str:
        section = arguments.get("section", "")

        # Check if already loaded
        for loaded_name in self.loaded_sections:
            if section.lower() == loaded_name.lower():
                return (
                    f'[SYSTEM] Policy "{loaded_name}" is already loaded. '
                    f"See \"POLICIES (previously loaded)\" in your system prompt."
                )

        # exact match
        if section in self.policy_sections:
            self.loaded_sections[section] = self.policy_sections[section]
            return f'[SYSTEM] Policy "{section}" loaded successfully. It is now available in your system prompt above.'

        # case-insensitive partial match
        section_lower = section.lower()
        for name, content in self.policy_sections.items():
            if section_lower in name.lower() or name.lower() in section_lower:
                self.loaded_sections[name] = content
                return f'[SYSTEM] Policy "{name}" loaded successfully. It is now available in your system prompt above.'

        available = ", ".join(f'"{n}"' for n in self.policy_sections if n != "_intro")
        return f'[SYSTEM] Policy "{section}" not found. Available sections: {available}'

    # ---- main loop ----

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        # First message: parse instructions and build system prompt
        if not self.conversation:
            instructions, user_text = input_text.split(MESSAGES_DELIMITER, 1)
            self._base_system_prompt = self._build_system_prompt(instructions)
            input_text = user_text.strip()

        self.conversation.append({"role": "user", "content": input_text})
        self._turn_count = getattr(self, "_turn_count", 0) + 1
        logger.info(
            "%s══════ Turn %d ══════%s", _CYAN, self._turn_count, _RESET
        )
        logger.info(
            "%sUSER:%s %s", _BOLD, _RESET, input_text[:200]
        )

        # Ephemeral working messages for internal loop (not persisted)
        working_messages = list(self.conversation)

        output = None
        for step in range(MAX_INTERNAL_STEPS):
            response_text = await self.llm.call(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    *working_messages,
                ],
            )

            # Parse LLM JSON
            parsed = self._parse_llm_output(response_text)
            if parsed is None:
                logger.info(
                    "  %s[step %d] ✗ PARSE ERROR%s → retry\n"
                    "  %sRaw: %s%s",
                    _RED, step, _RESET,
                    _DIM, response_text[:150], _RESET,
                )
                working_messages.append({"role": "assistant", "content": response_text})
                working_messages.append({"role": "user", "content": FORMAT_REMINDER})
                continue

            tool_name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            thinking = parsed.get("thinking", "")

            if thinking:
                logger.info(
                    "  %sthinking: %s%s", _DIM, thinking[:150], _RESET
                )

            # --- internal tool: lookup_policy ---
            if tool_name == "lookup_policy":
                section = arguments.get("section", "?")
                result = self._handle_lookup_policy(arguments)
                found = "not found" not in result.lower()
                logger.info(
                    "  %s[step %d] lookup_policy(\"%s\") → %s%s",
                    _YELLOW, step, section,
                    f"{len(result)} chars" if found else "NOT FOUND",
                    _RESET,
                )
                working_messages.append({"role": "assistant", "content": response_text})
                working_messages.append({
                    "role": "user",
                    "content": f"[INTERNAL TOOL RESULT: lookup_policy]\n{result}",
                })
                continue

            # --- domain tool or respond → final answer ---
            if tool_name == "transfer_to_human_agents":
                self._transfer_done = True

            output = json.dumps({"name": tool_name, "arguments": arguments})
            output_for_conv = output
            if thinking:
                output_for_conv = json.dumps({"thinking": thinking, "name": tool_name, "arguments": arguments})
            if tool_name == "respond":
                content = arguments.get("content", "")
                logger.info(
                    "  %s[step %d] ✓ RESPOND:%s %s",
                    _GREEN, step, _RESET, content[:150],
                )
            else:
                logger.info(
                    "  %s[step %d] ✓ TOOL:%s %s(%s)",
                    _MAGENTA, step, _RESET, tool_name,
                    json.dumps(arguments)[:120],
                )
            break
        else:
            logger.warning(
                "  %s[!] Max steps (%d) reached%s", _RED, MAX_INTERNAL_STEPS, _RESET
            )
            output = json.dumps({
                "name": "respond",
                "arguments": {"content": "I apologize, could you please repeat your request?"},
            })
            output_for_conv = output

        # Save output with thinking to persistent conversation
        self.conversation.append({"role": "assistant", "content": output_for_conv})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=output))],
            name="Response",
        )

    # ---- helpers ----

    @staticmethod
    def _parse_llm_output(text: str) -> dict | None:
        """Extract JSON from LLM output, handling markdown code fences."""
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        logger.warning("Failed to parse JSON from: %s", text[:200])
        return None
