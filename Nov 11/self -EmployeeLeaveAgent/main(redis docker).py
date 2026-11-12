
import os
import json

from uuid import UUID, uuid4

import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient


# STEP 1: Load configuration
load_dotenv()

app = FastAPI(title="Employee Leave Agent API")

SESSION_TTL = 3600  # 1 hour


# STEP 2: Connect to Redis
db = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True
)


# STEP 3: Connect to existing Foundry agent
project = AIProjectClient(
    endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
    allow_preview=True
)

agent = project.get_openai_client(
    agent_name=os.getenv("FOUNDRY_AGENT_NAME", "EmployeeLeaveAgent")
)


# STEP 4: Define request structure
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: UUID | None = None


# STEP 5: Chat API
@app.post("/chat")
def chat(request: ChatRequest):

    session_id = str(request.session_id or uuid4())

    conversation_key = f"employee:{session_id}:conversation"
    history_key = f"employee:{session_id}:history"

    # Get existing conversation from Redis
    conversation_id = db.get(conversation_key)

    # Create conversation for a new customer
    if not conversation_id:
        conversation = agent.conversations.create()
        conversation_id = conversation.id

        db.set(
            conversation_key,
            conversation_id,
            ex=SESSION_TTL
        )

    # Call existing Foundry agent
    response = agent.responses.create(
        conversation=conversation_id,
        input=request.message
    )

    answer = response.output_text or ""

    # Store chat history in Redis
    with db.pipeline() as pipe:
        pipe.rpush(
            history_key,
            json.dumps({
                "role": "user",
                "message": request.message
            }),
            json.dumps({
                "role": "assistant",
                "message": answer
            })
        )

        pipe.expire(history_key, SESSION_TTL)
        pipe.expire(conversation_key, SESSION_TTL)
        pipe.execute()

    return {
        "session_id": session_id,
        "reply": answer
    }


# STEP 6: Retrieve Redis chat history
@app.get("/history/{session_id}")
def get_history(session_id: UUID):

    history_key = f"employee:{session_id}:history"

    messages = db.lrange(history_key, 0, -1)

    return {
        "session_id": str(session_id),
        "messages": [
            json.loads(message)
            for message in messages
        ]
    }

