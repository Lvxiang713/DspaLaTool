"""Configurable Qwen/DeepSeek transport; no retries or redirects."""
import json
import os
import urllib.request
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor
from .capabilities import reasoning_capability


class ProviderError(Exception):
    """Only safe public metadata; never include response bodies or credentials."""
    def __init__(self, code, message, http_status=None):
        super().__init__(message)
        self.code, self.http_status = code, http_status

    def public(self):
        return {"code": self.code, "message": str(self), "http_status": self.http_status}


def environment_config():
    return {"provider": os.environ.get("TABULAR_AGENT_LLM_PROVIDER", "qwen"),
            "model": os.environ.get("TABULAR_AGENT_LLM_MODEL", ""),
            "base_url": os.environ.get("TABULAR_AGENT_LLM_BASE_URL", ""),
            "api_key": os.environ.get("TABULAR_AGENT_LLM_API_KEY", ""),
            "protocol": os.environ.get("TABULAR_AGENT_LLM_PROTOCOL", "chat_completions"),
            "reasoning_effort": os.environ.get("TABULAR_AGENT_LLM_REASONING", "off"),
            "allow_http": os.environ.get("TABULAR_AGENT_LLM_ALLOW_HTTP") == "1"}


class ChatProvider:
    def __init__(self, config=None, discovery=False):
        config = environment_config() if config is None else dict(config)
        self.family = str(config.get("provider", "")).lower()
        self.model = str(config.get("model", "")).strip()
        if self.family=="others":
            lowered=self.model.casefold()
            self.family="deepseek" if "deepseek" in lowered else "qwen" if "qwen" in lowered else "others"
        self.base = str(config.get("base_url", "")).strip().rstrip("/")
        self.key = str(config.get("api_key", ""))
        self.protocol = config.get("protocol", "chat_completions")
        self.effort = config.get("reasoning_effort", "off")
        self.capability=reasoning_capability(self.family,self.model)
        allowed=self.capability["levels"]
        if self.capability["kind"]=="fixed" and self.effort=="off":
            self.effort="on"
        if self.effort not in allowed:
            raise ValueError("Choose a supported reasoning level for this provider.")
        self.reply_limit=None
        if self.protocol not in ("chat_completions", "responses"):
            raise ValueError("Please select an API protocol.")
        if self.family not in ("qwen", "deepseek", "others") or not all((self.base, self.key)) or (not discovery and not self.model):
            raise ValueError("Complete the provider, model, API base URL and API key in LLM settings.")
        parsed = urlsplit(self.base)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query
                or parsed.fragment or len(self.base) > 2000 or len(self.model) > 200
                or len(self.key) > 8192 or any(c in self.key for c in "\r\n")):
            raise ValueError("Invalid LLM settings. Use an API base URL without credentials, query or fragment.")
        if not self.base.startswith("https://"):
            if not (self.base.startswith("http://") and
                    config.get("allow_http") is True):
                raise ValueError("Use HTTPS, or explicitly set TABULAR_AGENT_LLM_ALLOW_HTTP=1 for your trusted HTTP endpoint.")
        self.calls = 0
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _open(self, path, data=None, timeout=None):
        req = urllib.request.Request(self.base + path,
            data=None if data is None else json.dumps(data).encode("utf-8"),
            headers={"Authorization":"Bearer "+self.key, "Content-Type":"application/json"})
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        try:
            return urllib.request.build_opener(NoRedirect).open(req, timeout=timeout)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            messages = {401:"Authentication failed. Check the API key.",
                        403:"Endpoint denied access.",
                        404:"Model or API route not found. Use Discover models to check the exact model ID.",
                        400:"Endpoint rejected the request. Check model ID and protocol.",
                        429:"Endpoint is busy or rate-limited. Try again later."}
            raise ProviderError("http_error", messages.get(status,"The endpoint returned an HTTP error."),status) from None
        except (URLError, OSError):
            raise ProviderError("connection_error","Could not reach the endpoint. Check its address and network.") from None

    def models(self):
        with self._open("/models",timeout=15) as response:
            obj=json.loads(response.read(1000000))
        return [x["id"] for x in obj.get("data",[]) if isinstance(x.get("id"),str)][:200]

    def _request(self, instructions, payload, max_tokens, timeout, on_delta=None):
        # Intent selection and connection checks stay fast. Selected reasoning
        # applies only to the final, streamed answer.
        effort=self.effort if on_delta is not None or self.capability["kind"]=="fixed" else "off"
        native_effort=("xhigh" if self.family=="qwen" and
                       self.model.lower().startswith("qwen3.8") and effort=="high" else effort)
        if effort!="off":
            budgets=({"on":6144,"low":3072,"medium":6144,"high":8192} if self.family=="qwen"
                     else {"on":8192,"low":4096,"high":16384,"max":32768})
            max_tokens=max(max_tokens,budgets[effort])
        if self.protocol == "responses":
            path = "/responses"
            data = {"model": self.model, "instructions": instructions,
                    "input": json.dumps(payload, ensure_ascii=False),
                    "stream": on_delta is not None, "max_output_tokens": max_tokens, "store": False}
        else:
            path = "/chat/completions"
            data = {"model": self.model, "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "stream": on_delta is not None, "max_tokens": max_tokens}
        configurable=self.capability["kind"] in ("toggle","effort")
        if self.family == "qwen" and configurable:
            data["chat_template_kwargs"] = {"enable_thinking": effort!="off"}
        elif self.family=="deepseek" and configurable and self.protocol=="chat_completions":
            data["thinking"]={"type":"enabled" if effort!="off" else "disabled"}
        if self.capability["kind"]=="effort" and (effort!="off" or self.family=="deepseek" and self.protocol=="responses"):
            if self.protocol=="responses":
                data["reasoning"]={"effort":"none" if effort=="off" else native_effort}
            elif effort!="off":
                data["reasoning_effort"]=native_effort
        with self._open(path,data,timeout) as response:
            if on_delta is not None:
                chunks=[]
                size=0
                complete=False
                for raw_line in response:
                    size+=len(raw_line)
                    if size>2000000:
                        raise ProviderError("response_size","Endpoint stream exceeded the response size limit.")
                    line=raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    content=line[5:].strip()
                    if content=="[DONE]":
                        complete=True
                        break
                    event=json.loads(content)
                    if event.get("error") or event.get("type") in ("error","response.failed","response.error"):
                        raise ProviderError("generation_failed","The endpoint could not complete generation.")
                    delta=""
                    if self.protocol=="responses":
                        if event.get("type")=="response.output_text.delta":
                            delta=event.get("delta","")
                        if event.get("type")=="response.completed":
                            complete=True
                        if event.get("type")=="response.incomplete":
                            raise ProviderError("truncated","Response reached the output limit. Ask a shorter question.")
                    else:
                        choices=event.get("choices",[])
                        if choices:
                            delta=choices[0].get("delta",{}).get("content") or ""
                            finish=choices[0].get("finish_reason")
                            if finish=="length":
                                raise ProviderError("truncated","Response reached the output limit. Ask a shorter question.")
                            if finish=="stop":
                                complete=True
                    if isinstance(delta,str) and delta:
                        chunks.append(delta)
                        on_delta(delta)
                if not complete:
                    raise ProviderError("interrupted","Endpoint stream ended before completion.")
                text="".join(chunks).strip()
                if not text:
                    raise ProviderError("empty_response","The endpoint returned no answer text.")
                return text
            raw = response.read(1000001)
        if len(raw) > 1000000:
            raise ValueError("LLM response exceeded size limit.")
        obj = json.loads(raw)
        if obj.get("status")=="incomplete":
            raise ProviderError("truncated","The endpoint used the output budget before finishing. Try a shorter question.")
        if self.protocol == "responses":
            text = "\n".join(part.get("text", "") for item in obj.get("output", [])
                if item.get("type") == "message" for part in item.get("content", [])
                if part.get("type") == "output_text")
        else:
            text = obj["choices"][0]["message"].get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM returned no answer text.")
        return text

    def call(self, instructions, payload, max_tokens, on_delta=None):
        if self.calls >= 3:
            raise ProviderError("call_limit","Assistant model-call limit reached.")
        self.calls += 1
        future = self.pool.submit(self._request, instructions, payload, max_tokens, None, on_delta)
        return future.result()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
