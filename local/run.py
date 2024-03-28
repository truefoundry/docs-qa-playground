from backend.modules.query_controllers.example.types import DefaultQueryInput
from backend.modules.query_controllers import ExampleQueryController
from backend.modules.query_controllers.example.payload import QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY
import asyncio

# You can change the query here
print(f"Default Query: {QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY}")

# Create a request object
request = DefaultQueryInput(**QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY)

# Create a controller object
controller = ExampleQueryController()

# Get the answer
answer = asyncio.run(controller.answer(request))
print(f"Answer: {answer}")