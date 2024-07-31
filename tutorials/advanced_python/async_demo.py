import asyncio
import time
from datetime import datetime
from decorator import decorator
from typing import Optional


@decorator
async def async_fn_log(func, *args, **kwargs):
    """Logs the function call"""
    time_start = time.time()
    print(f"# {datetime.fromtimestamp(time_start)}: Calling {func.__name__}(args={args}, kwargs={kwargs})")
    return_values = await func(*args, **kwargs)
    time_end = time.time()
    print(f"# {datetime.fromtimestamp(time_end)}: Func {func.__name__} Returning {return_values} in {time_end - time_start:.2f}s")
    return return_values

@async_fn_log
async def async_query(query:str) -> str:
    await asyncio.sleep(1)
    return f"Query result for {query=}"

@async_fn_log
async def async_chat(message:str) -> str:
    await asyncio.sleep(3)
    return f"Chat result for {message=}"

@async_fn_log
async def async_update(key:str, value:object) -> Optional[str]:
    await asyncio.sleep(2)
    return f"Update result for {key=}, {value=}"

@async_fn_log
async def async_upload(data:str) -> Optional[str]:
    await asyncio.sleep(1)
    return f"Upload result for {data=}"

@async_fn_log
async def async_download(url:str) -> Optional[str]:
    await asyncio.sleep(1)
    return f"Download result for {url=}"

@async_fn_log
async def async_buessiness_logic():
    user_info, item_info = await asyncio.gather(
        async_query("user_id"),
        async_query("item_id"),
    )
    user_data, item_data = await asyncio.gather(
        async_download(item_info),
        async_download(user_info),
    )
    await asyncio.gather(
        async_chat(user_data + item_data),
        async_upload(f"{user_data=}, {item_data=}")
    )
    await asyncio.gather(
        async_update("user_data", user_data),
        async_update("item_data", item_data),
    )

@async_fn_log
async def main():
    return await asyncio.gather(
        async_query("query1"),
        async_query("query2"),
        async_chat("message1"),
        async_chat("message2"),
        async_update("key1", "value1"),
        async_update("key2", "value2"),
        async_upload("data1"),
        async_upload("data2"),
        async_download("url1"),
        async_download("url2"),
    )

if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(async_buessiness_logic())