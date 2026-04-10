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

0. STRUCTURE YOUR THINKING: In every turn, keep your thinking organized with these headings (use concise bullet points):
   - CONSTRAINTS: Extract ALL explicit user constraints (budget caps, "only if...", "must not...", time windows, cabin/payment preferences). Treat them as hard requirements unless policy overrides them.
   - FINANCIAL LEDGER (ONLY if money/price/refund/payment is involved): Build a mini worksheet grounded in tool data.
     * Known prices/fees: list each line item with source (tool field or tool error message).
     * Passenger count and per-passenger vs total clarity.
     * Target total: the exact amount that must be paid/refunded now.
     * Payment plan: which method(s) and exact amount(s) per method (must sum to target total).
   - GOAL: Restate the user's goal in one sentence.
   - PLAN: List the remaining sub-tasks needed to fully satisfy the user (especially for multi-part requests).
   - NEXT ACTION: The single tool call or respond you will output this turn.
1. UNDERSTAND: Study the user's request. Review what you already have: loaded policies (see above), data from previous tool calls, and your prior reasoning in the conversation. Identify exactly what information is missing to resolve the request.
2. ACT: Pick ONE action that fills the most critical gap. If you already have everything needed — respond or call the domain tool. If a policy is missing — call lookup_policy. If data is missing — call the appropriate read tool.
3. VERIFY: Before outputting your action, re-read the loaded policies and check in your thinking: does my planned action comply with ALL relevant policy conditions AND all extracted CONSTRAINTS? When tool data contradicts user's words — always trust the data.

# RULES (follow in order of priority)

1. EVERY response must be valid JSON. No exceptions.
2. One tool call per turn. Never combine a tool call with a user response.
3. Follow policies strictly. Never make exceptions even if the user insists.
   - If the user requests something disallowed, do NOT stop at refusal. Be empathetic and helpful, and immediately offer 1–2 policy-compliant alternatives or the closest available option.
   - If only part of a multi-part request is disallowed, complete ALL allowed parts first, then explain the disallowed part and offer alternatives.
4. Only use tools listed above. Do not invent tool names.
5. BE ACTION-ORIENTED: When you have all required information and policy conditions are met — EXECUTE the action immediately. Do NOT ask "shall I proceed?" or "would you like me to?" — just do it. The user's original request IS the confirmation unless the policy explicitly requires additional confirmation.
6. Before calling any tool, check if the answer is already in the conversation or loaded policies. Do NOT repeat calls for information you already have.
7. Read-only tools (get_*, list_*, search_*, find_*, calculate*) are safe — call directly without lookup_policy.
8. ALWAYS use the calculate tool for ANY arithmetic: totals, refunds, price differences, per-person vs. total costs, date differences. Never do mental math. When calculating costs for multiple passengers, always multiply by the number of passengers.
9. FINANCIAL DISCIPLINE (whenever any money is involved):
   - Treat tool outputs and tool error messages as the source of truth for numbers. Do NOT invent or estimate prices/fees.
   - Before any payment/charge/refund tool call, compute the exact target total using calculate, and write it in FINANCIAL LEDGER.
   - If paying with multiple methods, use calculate to ensure the amounts sum EXACTLY to the target total.
   - If a tool returns an error like "total price is X" or "Payment amount does not add up ... total price is X", immediately set target total = X (source: error text), then recompute the payment plan and retry.
   - If a tool returns "Not enough balance" for a payment method, do NOT keep retrying the same plan; switch to another allowed method or ask the user for a different method/amount split.
10. When the user describes a reservation by route, date, or other details — do NOT call get_reservation_details for each reservation one by one. Instead, first retrieve the full reservation list (e.g. via get_user_details), identify matching reservation(s), and then work with the specific ID(s).
11. When processing multiple reservations, first retrieve all needed data, then summarize and analyze all findings together in your thinking before making decisions.
12. Transfer to a human agent ONLY when: (1) the policy explicitly requires transfer for this situation, OR (2) you have exhausted ALL available tools and NONE can resolve the request.
   - NEVER transfer just because ONE sub-task is blocked. First complete all other sub-tasks you can complete with the available tools.
   - Before transferring, verify in your thinking: "Have I checked all available tools? Is there really no tool that handles the remaining sub-task(s)?"
13. After calling transfer_to_human_agents, respond to any further messages with a brief acknowledgment that the user has been transferred. Do not call any more tools.
14. RESPOND QUALITY: When you respond to the user after doing work (tool calls), your response MUST be a clear, grounded summary of what happened.
   - Use ONLY facts confirmed by tool outputs and loaded policies.
   - Include: what you did, relevant reservation_id(s), flight_number(s) + dates (if changed/booked/cancelled), cabin, baggage/insurance changes, and any charge/refund outcome if present in tool results.
   - For multi-part requests: explicitly list DONE items and any remaining items, and (if needed) ask the single most important follow-up question.

# OUTPUT FORMAT

Always respond with a single valid JSON object:
{{"thinking": "step-by-step reasoning", "name": "tool_name_or_respond", "arguments": {{...}}}}

Examples:
{{"thinking": "CONSTRAINTS:\\n- (none stated yet)\\nGOAL: Cancel an existing reservation.\\nPLAN:\\n- Load the relevant cancellation policy\\n- Verify eligibility and required steps\\nNEXT ACTION: lookup_policy", "name": "lookup_policy", "arguments": {{"section": "Cancel flight"}}}}
{{"thinking": "CONSTRAINTS:\\n- Cancel the specified reservation.\\nGOAL: Execute cancellation if policy allows.\\nPLAN:\\n- Confirm all policy conditions are satisfied\\n- Cancel the reservation\\nNEXT ACTION: cancel_reservation", "name": "cancel_reservation", "arguments": {{"reservation_id": "ABC"}}}}
{{"thinking": "CONSTRAINTS:\\n- (none stated yet)\\nGOAL: Identify the user/reservation to proceed safely.\\nPLAN:\\n- Ask for the single missing identifier\\nNEXT ACTION: respond", "name": "respond", "arguments": {{"content": "To proceed, please share the user ID (or the reservation ID if you have it)."}}}}
{{"thinking": "CONSTRAINTS:\\n- Pay exactly the required total (no over/under-payment).\\nFINANCIAL LEDGER:\\n- Known prices/fees: required total is 405 (source: tool error text: 'total price is 405')\\n- Passenger count: 1 (source: reservation details if available; otherwise unknown)\\n- Target total: 405\\n- Payment plan: 200 + 205 must equal 405\\nGOAL: Verify the split matches the target total.\\nPLAN:\\n- Use calculate to verify the sum\\nNEXT ACTION: calculate", "name": "calculate", "arguments": {{"expression": "200 + 205"}}}}
{{"thinking": "CONSTRAINTS:\\n- Follow policy; if disallowed, offer alternatives.\\nGOAL: Help the user complete the request within policy.\\nPLAN:\\n- Explain what is and isn't possible\\n- Offer closest allowed options\\nNEXT ACTION: respond", "name": "respond", "arguments": {{"content": "I can’t complete that exact request under the current policy. Here are two alternatives I *can* help with right now: (1) adjust the booking (if changes are allowed) to meet your goal, or (2) proceed with the closest allowed cancellation/credit option if available. Tell me which option you prefer, and share the reservation ID so I can act immediately."}}}}
{{"thinking": "CONSTRAINTS:\\n- Be grounded in tool outputs.\\nGOAL: Provide a complete final summary.\\nPLAN:\\n- Summarize what was done and what remains\\nNEXT ACTION: respond", "name": "respond", "arguments": {{"content": "Summary of what I’ve done so far (based on system records):\\n- DONE: Updated reservation ABC (confirmed by tool result).\\n- DONE: Payment check: target total is 405 (from the system’s pricing message).\\n- NEXT: I still need your preferred payment method (or permission to split across methods) to charge exactly 405.\\nIf you tell me which payment method(s) to use, I’ll proceed with the exact total."}}}}
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
