import argparse
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Silence noisy loggers
for _noisy in ("httpx", "httpcore", "litellm", "openai", "LiteLLM",
               "LiteLLM Proxy", "LiteLLM Router", "uvicorn.access"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Suppress litellm's print-based spam
import litellm
litellm.suppress_debug_info = True

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()
    
    skill = AgentSkill(
        id="customer-service",
        name="Customer Service Agent",
        description="Handles customer service tasks across airline, retail, and telecom domains using tools and policies",
        tags=["customer-service", "tool-use"],
        examples=["Help me cancel my reservation", "I need to change my flight"],
    )

    agent_card = AgentCard(
        name="Customer Service Agent",
        description="Customer service agent that handles tasks across airline, retail, and telecom domains using tools and policies",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
