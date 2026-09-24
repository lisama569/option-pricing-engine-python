"""本地启动入口：uvicorn main:app。"""

from option_engine.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
