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

# AVAILABLE TOOLS

Domain tools you can call (one per turn):
{tools_json}

You can also respond to the user directly:
{{"thinking": "...", "name": "respond", "arguments": {{"content": "your message"}}}}

Internal tool (invisible to user — use to review policy before acting):
{{"thinking": "why I need this section", "name": "lookup_policy", "arguments": {{"section": "section name"}}}}

# HOW TO WORK — PAR (Plan → Act → Reflect)

1. PLAN: Think about what the user needs and which policy section applies.
2. ACT: If unsure about rules — call lookup_policy FIRST. If you know the rules — call the appropriate domain tool or respond.
3. REFLECT: After getting a tool result, reason about what you learned and what to do next.

Rules:
- ALWAYS call lookup_policy as your FIRST action on every new user request. Never skip this step — you must check the relevant policy section before calling any domain tool or responding with information.
- You can call lookup_policy multiple times if you need additional sections during the same request.
- Read-only tools (get_*, list_*, search_*, find_*, calculate*) are safe — call directly.
- BE ACTION-ORIENTED: When the user asks you to do something (cancel, book, change, update, etc.) and you have gathered all required information and policy conditions are met — EXECUTE the action by calling the tool. Do NOT just describe what you could do. Your job is to complete the task, not to explain it.
- When the user's intent is clear from their message (e.g. "cancel my reservation", "change my flight"), treat that as confirmation of intent. Only ask for additional confirmation if the policy specifically requires it or if there are significant consequences the user might not be aware of (e.g., non-refundable fees).
- Follow policies strictly. Never make exceptions even if the user insists.
- Only use tools listed above. Do not invent tool names.
- One tool call per turn. Never combine a tool call with a user response.
- Transfer to a human agent ONLY as a last resort when you truly cannot handle the request with available tools. If you have tools that can solve the problem — use them instead of transferring. After calling transfer_to_human_agents, the conversation is OVER — do not call it again.
- EVERY response must be valid JSON. No exceptions, even after transfer.

# OUTPUT FORMAT

Always respond with a single valid JSON object:
{{"thinking": "step-by-step reasoning", "name": "tool_name_or_respond", "arguments": {{...}}}}

Examples:
{{"thinking": "User wants to cancel. I need to check cancellation rules first.", "name": "lookup_policy", "arguments": {{"section": "Cancel flight"}}}}
{{"thinking": "Policy says X. User confirmed. All conditions met.", "name": "cancel_reservation", "arguments": {{"reservation_id": "ABC"}}}}
{{"thinking": "I need the user's ID to proceed.", "name": "respond", "arguments": {{"content": "Could you provide your user ID?"}}}}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.llm = LLMClient()
        self.system_prompt: str = ""
        self.conversation: list[dict] = []  # persistent across turns
        self.policy_sections: dict[str, str] = {}
        self._transfer_done: bool = False

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
        )

    # ---- internal tools ----

    def _handle_lookup_policy(self, arguments: dict) -> str:
        section = arguments.get("section", "")

        # exact match
        if section in self.policy_sections:
            return f"=== Policy: {section} ===\n{self.policy_sections[section]}"

        # case-insensitive partial match
        section_lower = section.lower()
        for name, content in self.policy_sections.items():
            if section_lower in name.lower() or name.lower() in section_lower:
                return f"=== Policy: {name} ===\n{content}"

        available = ", ".join(f'"{n}"' for n in self.policy_sections if n != "_intro")
        return f"Section '{section}' not found. Available sections: {available}"

    # ---- main loop ----

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        # First message: parse instructions and build system prompt
        if not self.conversation:
            instructions, user_text = input_text.split(MESSAGES_DELIMITER, 1)
            self.system_prompt = self._build_system_prompt(instructions)
            input_text = user_text.strip()

        # If transfer was already called, auto-respond without LLM
        if self._transfer_done:
            logger.info(
                "%s══════ Turn (post-transfer, auto) ══════%s", _CYAN, _RESET
            )
            output = json.dumps({
                "name": "respond",
                "arguments": {"content": "You have been transferred to a human agent who will assist you further."},
            })
            self.conversation.append({"role": "user", "content": input_text})
            self.conversation.append({"role": "assistant", "content": output})
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=output))],
                name="Response",
            )
            return

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
                    {"role": "system", "content": self.system_prompt},
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

        # Save only the final action to persistent conversation
        self.conversation.append({"role": "assistant", "content": output})

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
