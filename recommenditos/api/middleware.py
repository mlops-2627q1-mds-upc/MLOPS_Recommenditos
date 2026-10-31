"""The ASGI middlewares every request passes: its ID, a last-resort 500, the body limit.

Both are plain ASGI rather than Starlette's `BaseHTTPMiddleware`, which buffers
the request and response and is the slower and less predictable of the two.
"""

from http import HTTPStatus
import time
import uuid

from loguru import logger
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from recommenditos.api.errors import (
    BLANK,
    PAYLOAD_TOO_LARGE,
    PayloadTooLarge,
    ProblemError,
    problem_response,
    too_large_detail,
)
from recommenditos.api.metrics import route_template

REQUEST_ID_HEADER = "X-Request-ID"

#: NFR-09: request bodies are limited to 10 KB. The largest real listing as a
#: valuation request is 3.5 KB, and the length bounds of the request schemas keep
#: a body with every field at its bound under the limit.
MAX_BODY_BYTES = 10 * 1024


class RequestIdMiddleware:
    """Give every request an ID: in `request.state.request_id`, the header and the log.

    The ID is generated here, never taken from the client, so that it identifies
    one request of this service. `/predict` returns it in its body (FR-06) and
    every problem names it in `instance`. Every log line written while the request
    is answered carries it as `request_id` (`recommenditos.api.logs`), and the
    last one says how the request was answered: its method, route template,
    status and duration. That line replaces uvicorn's access log, so the service
    log holds no client address and no path the client chose (NFR-08); nginx's
    access log, which names the client, joins it by the `X-Request-ID` header.
    `request.state.request_started` is the `time.perf_counter()` at which the
    request arrived, which the prediction log measures the latency from (FR-13).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["request_started"] = started = time.perf_counter()
        # A request whose app raised before it answered gets uvicorn's 500.
        status = HTTPStatus.INTERNAL_SERVER_ERROR.value

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        with logger.contextualize(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            finally:
                # The route template, as the metrics count it, never the raw path
                # or its query string, which the client chooses and could hold
                # anything, an email address included (NFR-08).
                logger.info(
                    "{method} {path} {status} {duration_ms} ms",
                    method=scope["method"],
                    path=route_template(scope),
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )


class UnexpectedErrorMiddleware:
    """Answer an exception no handler took with a 500 problem, inside the request ID.

    Starlette's own catch-all sits outside every middleware, so its response would
    carry no `X-Request-ID`; this one sits inside `RequestIdMiddleware`. The cause
    goes to the log, not to the client.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as error:
            if started:
                raise
            request_id = str(scope.get("state", {}).get("request_id", ""))
            logger.opt(exception=error).error("the request failed")
            response = problem_response(
                request_id,
                ProblemError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "The server failed to answer this request. Quote the instance when you "
                    "report it.",
                ),
            )
            await response(scope, receive, send)


class BodySizeLimitMiddleware:
    """Refuse a body over `limit` bytes before it is read (NFR-09).

    A request that declares a larger `Content-Length` is answered with a 413 at
    once, and its body is never read. A request without one (`Transfer-Encoding:
    chunked`) cannot be judged up front, so its body is counted as it arrives and
    the read stops with a 413 as soon as the count passes the limit: at most one
    chunk beyond the limit is ever read.
    """

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            refusal = self._refusal(declared)
            if refusal is not None:
                status, type_, detail = refusal
                request_id = str(scope.get("state", {}).get("request_id", uuid.uuid4()))
                response = problem_response(request_id, ProblemError(status, detail, type_=type_))
                await response(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise PayloadTooLarge(self.limit)
            return message

        await self.app(scope, counting_receive, send)

    def _refusal(self, declared: str) -> tuple[int, str, str] | None:
        """The status, problem type and detail to refuse `declared` with, or None."""
        try:
            length = int(declared)
        except ValueError:
            return HTTPStatus.BAD_REQUEST, BLANK, "The Content-Length header is not a number."
        if length > self.limit:
            return (
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                PAYLOAD_TOO_LARGE,
                too_large_detail(self.limit),
            )
        return None
