import json
import logging
import copy
import jsonpatch

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

# How many recent messages the main agent sees (rest is in working memory)
CONTEXT_WINDOW = 10

# Sections whose content is always included in the system prompt
ALWAYS_ON_SECTIONS = {"_intro", "Domain Basic"}

# Structured output schema for memory manager (JSON Patch)
RESPONSE_FORMAT = {"type": "json_object"}

MAX_REPAIR_ATTEMPTS = 2

# Format hint injected when the model produces wrong structure
FORMAT_REMINDER = (
    "[SYSTEM] Your previous output was missing required fields. "
    "You MUST respond with a JSON object containing these keys:\n"
    '{"thinking": "your reasoning", "name": "tool_name_or_respond", "arguments": {...}}'
)


# ---------------------------------------------------------------------------
# Policy parsing
# ---------------------------------------------------------------------------

def parse_policy_sections(policy_text: str) -> dict[str, str]:
    """Split markdown policy into sections by ## headers."""
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
    """Recursively remove noise fields from JSON Schema."""
    schema.pop("title", None)
    schema.pop("additionalProperties", None)

    defs = schema.pop("$defs", None)

    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj and defs:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    return resolve_refs(_clean_schema(dict(defs[ref_name])))
            if "anyOf" in obj:
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
    """Extract the JSON tool array from the eval agent's tools+format block."""
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
# System prompt (main agent)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a customer service agent. Follow the policies strictly.

# POLICIES (always loaded)

{always_on_policy}

# POLICY SECTIONS AVAILABLE ON DEMAND (call lookup_policy to read):

{on_demand_list}

{loaded_policies_block}# WORKING MEMORY (maintained automatically — your source of truth for all retrieved data)

{working_memory}

# AVAILABLE TOOLS

Domain tools you can call (one per turn):
{tools_json}

You can also respond to the user directly:
{{"thinking": "...", "name": "respond", "arguments": {{"content": "your message"}}}}

Internal tool (invisible to user — use to review policy before acting):
{{"thinking": "why I need this section", "name": "lookup_policy", "arguments": {{"section": "section name"}}}}

# HOW TO WORK

1. UNDERSTAND: Study WORKING MEMORY and the latest message. What does the user need? What is missing?
2. PLAN: What steps are needed to reach the goal? If a plan exists in memory — follow it or adjust based on new information.
3. ACT: Execute ONE action — call a tool to gather or change data, OR respond to the user.
4. RE-EVALUATE: After receiving a result, check if the plan still works. If new data changes the situation — adjust.

# RULES

1. EVERY response must be valid JSON. No exceptions.
2. Only use tools listed above. Do not invent tool names. One tool call per turn.
3. USER ID: A user_id is a system identifier (e.g. "emma_kim_9957"), NEVER a person's name. If you only know the user's name — ask them for their user_id before calling get_user_details.
4. BEFORE ANY MUTATION (book, cancel, update, charge, refund):
   - Is user identity verified (USER section has real data, not "unknown")? If not — call get_user_details.
   - Is the relevant policy section loaded? If not — call lookup_policy.
   - Are all required data points present (reservation details, flight options, prices)? If not — gather them.
   - Does the action comply with loaded policies? When tool data contradicts user's words — trust the data.
   - The API does NOT validate policy rules — it will execute even invalid requests. YOU must verify all conditions.
   Then list the action details to the user and wait for their explicit confirmation before executing.
   Read-only tools (get_*, search_*, calculate*) do NOT require this gate — call them freely.
5. USE AVAILABLE DATA: Before asking the user or calling a tool, check if the answer is already in WORKING MEMORY or loaded policies. Do NOT repeat calls for information you already have.
6. ALWAYS use the calculate tool for ANY arithmetic. Never do mental math. When calculating costs for multiple passengers, always multiply by the number of passengers.
7. FINANCIAL: Before any payment, read FINANCIAL LEDGER in memory. Verify target_total with calculate. If a payment method has insufficient balance — switch to another method from USER.payment_methods immediately.
8. USER'S OWN DATA: When the user references "their" booking/flight/trip — look in their account (get_user_details → reservations → get_reservation_details), NOT the flight catalog. When the user describes a reservation by route or date — retrieve ALL reservations first, then match.
9. EXHAUSTIVE SEARCH: Check PLAN in memory for completed and pending steps.
   - When searching flights: ALWAYS try both direct AND one-stop. If direct returns empty — search one-stop before responding.
   - When checking reservations: check ALL reservations in the user's list. Never conclude "not found" until every single one has been retrieved.
   - When processing multiple items: retrieve ALL first, then analyze together. Never decide from partial results.
10. ERROR RECOVERY: When a tool returns an error, read it carefully — errors often contain the correct value or a hint on how to retry. Extract the corrected data and retry. Do NOT abandon the goal.
11. ALTERNATIVES: If policy blocks an action, consider indirect paths (cancel + rebook, upgrade then act, split into steps). Always offer the user a viable next step.

# RESPONDING TO THE USER

- Ground EVERY claim in data from WORKING MEMORY. Include all relevant IDs, numbers, amounts, payment details.
- For financial operations: state exact amount, payment method, and policy basis.
- For multi-part requests: check PLAN in memory — list completed items and remaining items explicitly.
- If the user requests something disallowed: be empathetic, explain why, and offer 1–2 policy-compliant alternatives based on tool data (never offer compensation unless the user explicitly asks).
- If only part of a multi-part request is disallowed: complete ALL allowed parts first, then explain the disallowed part.

# TRANSFER TO HUMAN

- Transfer ONLY when: (1) the policy explicitly requires it, OR (2) you have exhausted ALL available tools and NONE can resolve the request.
- NEVER transfer just because ONE sub-task is blocked. First complete all other sub-tasks you can handle.
- Before transferring, verify in your thinking: "Have I checked all available tools? Is there really no tool that handles the remaining sub-task(s)?"

# OUTPUT FORMAT

Always respond with a single valid JSON object:
{{"thinking": "justify your action: what you know (from WORKING MEMORY), what is missing, why this action is the best next step", "name": "tool_name_or_respond", "arguments": {{...}}}}

Examples:
{{"thinking": "User wants to cancel. Memory has reservation but policy not loaded — need lookup before cancel.", "name": "lookup_policy", "arguments": {{"section": "Cancel flight"}}}}
{{"thinking": "Policy loaded, user verified, all conditions met per policy. Executing cancellation.", "name": "cancel_reservation", "arguments": {{"reservation_id": "ABC"}}}}
{{"thinking": "Memory shows user not verified. Cannot proceed without user_id.", "name": "respond", "arguments": {{"content": "Could you provide your user ID so I can look up your account?"}}}}
{{"thinking": "Need to verify payment split before charging. 200 + 205 = ?", "name": "calculate", "arguments": {{"expression": "200 + 205"}}}}
"""


# ---------------------------------------------------------------------------
# Memory manager prompt
# ---------------------------------------------------------------------------

MEMORY_MANAGER_PROMPT = """\
You maintain WORKING MEMORY for a customer service agent via JSON Patch operations (RFC 6902).

CURRENT MEMORY (JSON):
{current_memory}

NEW EVENTS:
{new_events}

# MEMORY SCHEMA

The memory has 5 sections. Here is what each contains and the expected data format:

## /user — Verified customer profile
- id: string — user ID (e.g. "usr_12345"). Source: get_user_details.
- name: string — full name. Source: get_user_details.
- membership: string — tier level (e.g. "gold", "silver", "regular"). Source: get_user_details.
- payment_methods: array of objects — each: {{"id": "pm_1", "type": "credit_card", "brand": "visa", "last4": "1234", "balance": null}} or {{"id": "pm_2", "type": "gift_card", "balance": 150.0}}. Source: get_user_details.
- saved_passengers: array of objects — each: {{"name": "John Doe", "dob": "1990-01-15"}}. Source: get_user_details.
- reservations: array of strings — reservation IDs. Source: get_user_details.

## /request — What the customer wants
- original: string — first request verbatim essence. Set once, never changed.
- current: string — latest version of the request (may evolve as user refines).
- constraints: array of strings — explicit conditions: budget, dates, cabin class, preferences (e.g. "budget max $500", "direct flights only", "window seat").

## /retrieved_data — Facts from tool calls (free-form dict)
Keys are descriptive IDs (e.g. "reservation_ABC", "flights_NYC_LAX", "calc_total").
Values are objects with extracted data — exact numbers, IDs, statuses. Never paraphrase.
Example: {{"reservation_XYZ": {{"status": "confirmed", "origin": "LAX", "destination": "JFK", "price": 350, "passengers": ["John Doe"]}}}}

## /financial_ledger — Precise accounting for current operation
- prices: object — item_key → amount (e.g. {{"flight_ABC": 250, "seat_upgrade": 50}}). Source: tool results.
- passenger_count: integer — number of passengers in current operation.
- target_total: string or number — calculated total with source (e.g. "500 (2 × 250, from calculate)"). Use "not yet calculated" if unknown.
- payment_plan: object — payment_method_id → amount (e.g. {{"pm_gift": 150, "pm_cc": 350}}).
- charged: array of strings — completed charges (e.g. "charged $150 to pm_gift").
- refunded: array of strings — completed refunds (e.g. "refunded $200 to pm_cc").

## /plan — Agent's action plan
- done: array of strings — completed steps (e.g. "get_user_details → verified usr_123").
- failed: array of strings — "action → error" to avoid repeating (e.g. "charge pm_2 $500 → insufficient balance").
- pending: array of strings — remaining steps in order.

# EVENT FORMAT

ASSISTANT messages are JSON: {{"thinking": "...", "name": "tool_or_respond", "arguments": {{...}}}}
- "name" is the tool called (or "respond" for user-facing replies)
- The next USER message after a tool call contains the tool's result
- Extract data from tool results into /retrieved_data with exact values
- IMPORTANT: Empty results ([], {{}}, "not found", errors) are also data — record them in /retrieved_data (e.g. "search_SFO_JFK_direct_0518": "no flights found") and add the step to /plan/done. This prevents the agent from re-searching or missing that a search was already performed.

# DATA SOURCE RULES

- /user fields (name, membership, payment_methods, saved_passengers, reservations):
  ONLY update from get_user_details tool results. NEVER from user's words.
  Exception: /user/id may be set from user's message when they provide their user_id.
- /retrieved_data: ONLY from tool results. Never from user claims.
- /plan/done: ONLY add a step when its tool result is VISIBLE in the events.
  If events show the user providing their user_id — that does NOT mean get_user_details was called.
- /plan/pending: ONLY update when the agent's thinking (in ASSISTANT message)
  explicitly lists next steps. On the first turn (no ASSISTANT message yet),
  leave pending as-is. NEVER generate plan steps from the user's request alone —
  planning is the agent's job, not yours.
- /request: Extract from user's words. This is the user's intent — record it faithfully.
- /financial_ledger: ONLY from tool results and calculate results.

# PATCH DESCRIPTION

Each operation is an object with:
- "op": "replace" | "add" | "remove"
- "path": JSON Pointer (e.g. "/user/id", "/retrieved_data/reservation_ABC")
- "value": new value (omit for "remove")

# PATCH RULES

- Only patch fields that CHANGED. If nothing changed, return [].
- Extract exact IDs, amounts, dates from tool outputs. Never round or paraphrase numbers.
- "replace": change an existing scalar or overwrite an entire array/object. Path must point to an existing key.
- "add": insert a new key into an object, OR append to an array using "/-" at the end of the path.
- "remove": delete an existing key or array element. Do NOT remove from an empty array.
- CRITICAL: "/-" (dash) can ONLY be used with "add" (append). NEVER use "/-" with "replace" or "remove".
- To update an entire array, use "replace" with the full array path (no dash): {{"op": "replace", "path": "/plan/done", "value": ["step1", "step2"]}}
- Do NOT output the full memory. Output ONLY patches for changed fields.

DO NOT INVENT ANY INFORMATION. JUST EXTRACT THE DATA FROM THE EVENTS.
You MUST respond with a JSON object: {{"patches": [...]}}

# EXAMPLE RESPONSES

EXAMPLE response with updates:
{{"patches": [
  {{"op": "replace", "path": "/user/id", "value": "usr_123"}},
  {{"op": "add", "path": "/retrieved_data/res_ABC", "value": {{"status": "confirmed", "price": 350}}}},
  {{"op": "add", "path": "/plan/done/-", "value": "get_user_details → verified usr_123"}},
  {{"op": "replace", "path": "/plan/pending", "value": ["look up cancellation policy", "cancel reservation"]}}
]}}

EXAMPLE response with no updates:
{{"patches": []}}
"""

INITIAL_MEMORY: dict = {
    "user": {
        "id": "unknown",
        "name": "unknown",
        "membership": "unverified",
        "payment_methods": [],
        "saved_passengers": [],
        "reservations": [],
    },
    "request": {
        "original": "awaiting first message",
        "current": "awaiting first message",
        "constraints": [],
    },
    "retrieved_data": {},
    "financial_ledger": {
        "prices": {},
        "passenger_count": 0,
        "target_total": "not yet calculated",
        "payment_plan": {},
        "charged": [],
        "refunded": [],
    },
    "plan": {
        "done": [],
        "failed": [],
        "pending": ["awaiting user request"],
    },
}


def render_memory_for_prompt(memory: dict) -> str:
    """Render JSON memory as compact readable text for the agent's system prompt."""
    lines = []
    for section, data in memory.items():
        title = section.upper().replace("_", " ")
        lines.append(f"## {title}")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and v:
                    lines.append(f"- {k}:")
                    for item in v:
                        lines.append(f"  - {item}")
                elif isinstance(v, list):
                    lines.append(f"- {k}: (none)")
                elif isinstance(v, dict) and v:
                    lines.append(f"- {k}:")
                    for sk, sv in v.items():
                        lines.append(f"  - {sk}: {sv}")
                elif isinstance(v, dict):
                    lines.append(f"- {k}: (none)")
                else:
                    lines.append(f"- {k}: {v}")
        else:
            lines.append(str(data))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.llm = LLMClient()
        self.memory_llm = LLMClient(models=["openrouter/openai/gpt-4.1-mini"])
        self._base_system_prompt: str = ""
        self.conversation: list[dict] = []  # full history (for memory manager)
        self.policy_sections: dict[str, str] = {}
        self.loaded_sections: dict[str, str] = {}
        self._working_memory: dict = copy.deepcopy(INITIAL_MEMORY)
        self._last_thinking_output: str | None = None

    # ---- setup ----

    def _build_system_prompt(self, instructions: str) -> str:
        policy_text, tools_part = instructions.split(TOOLS_DELIMITER, 1)

        self.policy_sections = parse_policy_sections(policy_text.strip())

        always_on_parts = []
        for name in ALWAYS_ON_SECTIONS:
            if name in self.policy_sections:
                always_on_parts.append(self.policy_sections[name])

        tools_json = extract_tools_json(tools_part)

        return SYSTEM_PROMPT_TEMPLATE.format(
            always_on_policy="\n\n".join(always_on_parts),
            on_demand_list="{on_demand_list}",
            tools_json=tools_json,
            loaded_policies_block="{loaded_policies_block}",
            working_memory="{working_memory}",
        )

    def _get_system_prompt(self) -> str:
        """Return system prompt with loaded policies and working memory."""
        prompt = self._base_system_prompt

        # Inject loaded policies
        if self.loaded_sections:
            loaded_block = "# POLICIES (previously loaded — do NOT call lookup_policy for these again)\n\n"
            for section_name, content in self.loaded_sections.items():
                loaded_block += f"## {section_name}\n{content}\n\n"
            prompt = prompt.replace("{loaded_policies_block}", loaded_block)
        else:
            prompt = prompt.replace("{loaded_policies_block}", "")

        # Inject on demand list
        on_demand_names = [
            n for n in self.policy_sections
            if n not in ALWAYS_ON_SECTIONS and n not in self.loaded_sections
        ]

        on_demand_list = "\n".join(f'- "{name}"' for name in on_demand_names)
        prompt = prompt.replace("{on_demand_list}", on_demand_list if on_demand_list else "(none)")

        # Inject working memory (render JSON dict as readable text)
        prompt = prompt.replace("{working_memory}", render_memory_for_prompt(self._working_memory))

        return prompt

    # ---- memory manager ----

    async def _repair_patches(
        self, broken: list[tuple[dict, str]], current_memory: dict
    ) -> list[dict]:
        """Ask LLM to fix specific broken patches."""
        issues = []
        for patch, error in broken:
            issues.append(f"Patch: {json.dumps(patch)}\nError: {error}")

        prompt = (
            "The following JSON Patch operations failed when applied to memory:\n\n"
            "CURRENT MEMORY:\n"
            f"{json.dumps(current_memory, indent=2)}\n\n"
            "FAILED PATCHES:\n" + "\n\n".join(issues) + "\n\n"
            "Fix each patch and return corrected patch operations."
        )
        try:
            result = await self.memory_llm.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format=RESPONSE_FORMAT,
            )
            return json.loads(result).get("patches", [])
        except Exception:
            return []

    async def _update_memory(self) -> None:
        """Call the memory manager LLM to produce JSON Patch ops and apply them."""

        # Construct new events for memory enhancement
        recent = self.conversation[-2:]
        events_parts = []
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            # Inject thinking into last assistant message for memory manager
            if role == "ASSISTANT" and self._last_thinking_output:
                content = self._last_thinking_output
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            events_parts.append(f"[{role}]: {content}")
        new_events = "\n\n".join(events_parts)

        # Get patches from memory manager
        prompt = MEMORY_MANAGER_PROMPT.format(
            current_memory=json.dumps(self._working_memory, indent=2),
            new_events=new_events,
        )
        messages = [{"role": "user", "content": prompt}]
        result = await self.memory_llm.call(
            messages=messages,
            temperature=0.0,
            response_format=RESPONSE_FORMAT,
        )
        logger.info("  %s[memory] raw: %s%s", _DIM, result[:120],_RESET)

        # Parse patches
        parsed = json.loads(result)
        patches = parsed.get("patches", [])
        if not patches:
            logger.info("  %s[memory] no changes%s", _DIM, _RESET)
            return
    
        # Apply patches one by one; collect failures for LLM repair
        current = self._working_memory
        applied = 0
        broken: list[tuple[dict, str]] = []  # (patch, error)
        for patch in patches:
            try:
                current = jsonpatch.apply_patch(current, [patch])
                applied += 1
            except jsonpatch.JsonPatchException as pe:
                broken.append((patch, str(pe)))

        # Attempt LLM repair for broken patches
        for _ in range(MAX_REPAIR_ATTEMPTS):
            if not broken:
                break
            repaired = await self._repair_patches(broken, current)
            broken = []
            for patch in repaired:
                try:
                    current = jsonpatch.apply_patch(current, [patch])
                    applied += 1
                except jsonpatch.JsonPatchException as pe2:
                    broken.append((patch, str(pe2)))

        if applied > 0:
            self._working_memory = current
            logger.info(
                "  %s[memory] snapshot:%s\n%s",
                _DIM, _RESET,
                render_memory_for_prompt(self._working_memory),
            )
        logger.info(
            "  %s[memory] applied %d out of %d patches%s",
            _DIM, applied, len(patches), _RESET,
        )

    # ---- internal tools ----

    def _handle_lookup_policy(self, arguments: dict) -> str:
        section = arguments.get("section", "")

        for loaded_name in self.loaded_sections:
            if section.lower() == loaded_name.lower():
                return (
                    f'[SYSTEM] Policy "{loaded_name}" is already loaded. '
                    f"See \"POLICIES (previously loaded)\" in your system prompt."
                )

        if section in self.policy_sections:
            self.loaded_sections[section] = self.policy_sections[section]
            return f'[SYSTEM] Policy "{section}" loaded successfully. It is now available in your system prompt above.'

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

        # Update working memory with new information
        await self._update_memory()

        # Build agent context: system prompt + recent messages only
        recent_messages = self.conversation[-CONTEXT_WINDOW:]
        working_messages = list(recent_messages)

        output = None
        for step in range(MAX_INTERNAL_STEPS):
            response_text = await self.llm.call(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    *working_messages,
                ],
                response_format=RESPONSE_FORMAT,
            )

            parsed = json.loads(response_text)
            if any(key not in parsed for key in ["name", "arguments"]):
                logger.info(
                    "  %s[step %d] ✗ BAD STRUCTURE%s → retry\n"
                    "  %sRaw: %s%s",
                    _RED, step, _RESET,
                    _DIM, response_text[:150], _RESET,
                )
                working_messages.append({"role": "assistant", "content": response_text})
                working_messages.append({"role": "user", "content": FORMAT_REMINDER})
                continue

            tool_name = parsed["name"]
            arguments = parsed.get("arguments", {})
            thinking = parsed.get("thinking", "")

            if thinking:
                logger.info(
                    "  %sthinking: %s%s", _DIM, thinking, _RESET
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
            output = json.dumps({"name": tool_name, "arguments": arguments})
            output_for_conv = output  # compact, no thinking — for main agent context
            # Save thinking version for memory manager
            self._last_thinking_output = (
                json.dumps({"thinking": thinking, "name": tool_name, "arguments": arguments})
                if thinking else output
            )
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

        # Save output to persistent conversation
        self.conversation.append({"role": "assistant", "content": output_for_conv})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=output))],
            name="Response",
        )

