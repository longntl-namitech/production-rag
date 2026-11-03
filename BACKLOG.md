# Backlog

## Timed-out cross-encoder keeps running

`asyncio.wait_for` cancels the await, not the worker thread, so a slow local rerank keeps a CPU busy after the request returned. Fixing it needs a process pool or a backend that can be cancelled.
