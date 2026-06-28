from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from apoptosis.core.session import (
    DEFAULT_DATA_DIR,
    configure_session,
    get_session_optional,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if get_session_optional() is None:
        configure_session(DEFAULT_DATA_DIR)
    yield


api = FastAPI(title="Apoptosis", version="0.1.0", lifespan=lifespan)

from apoptosis import routes  # noqa: E402, F401
