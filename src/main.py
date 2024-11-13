from helpers.oai import async_gpt_calls
import asyncio


async def main():
    start_time = asyncio.get_event_loop().time()
    responses = await async_gpt_calls(
        [f"Respond with the number {str(i)}" for i in range(100)]
    )
    for response in responses:
        print(str(response))
    print(f"Time taken: {asyncio.get_event_loop().time() - start_time}")


asyncio.run(main())
