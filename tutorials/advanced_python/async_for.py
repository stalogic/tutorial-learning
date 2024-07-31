import asyncio

async def async_data_generator():

    for i in range(10):
        yield await asyncio.sleep(1, i)

async def main():
    async for i in async_data_generator():
        print(i)



if __name__ == "__main__":
    asyncio.run(main())
    