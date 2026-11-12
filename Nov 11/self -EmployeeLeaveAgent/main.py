
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

# Connect to existing Microsoft Foundry project
project = AIProjectClient(
    endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
    allow_preview=True
)

# Connect to existing agent
client = project.get_openai_client(
    agent_name=os.getenv("FOUNDRY_AGENT_NAME", "EmployeeLeaveAgent")
)

# Create conversation
conversation = client.conversations.create()

print("Connected to EmployeeLeaveAgent")
print("Type exit to quit")

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    try:
        response = client.responses.create(
            conversation=conversation.id,
            input=question
        )

        print("\nAgent:", response.output_text)

    except Exception as error:
        print("Error:", error)
