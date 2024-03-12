import aiohttp
import asyncio

data = {
  "collection_name": "ps01",
  "retriever_config": {
    "search_type": "similarity",
    "k": 4,
    "fetch_k": 20,
    "filter": {}
  },
  "query": "What is credit card",
  "model_configuration": {
    "name": "openai-devtest/gpt-3-5-turbo",
    "parameters": {}
  },
  "prompt_template": "Here is the context information:\n\n'''\n{context}\n'''\n\nQuestion: {question}\nAnswer:"
}

async def test_streaming_response():
    url = 'http://localhost:8080/retrievers/stream'  # replace with your actual endpoint
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            async for chunk in resp.content.iter_chunks():
                print(chunk[0])

# Run the test
loop = asyncio.get_event_loop()
loop.run_until_complete(test_streaming_response())