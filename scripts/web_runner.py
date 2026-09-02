from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib import error, parse, request

from lib.benchlib import PowerSampler, atomic_json, call_performance_summary, distribution_summary, evaluate_checks, load_json, mode_to_think, response_metrics, utc_now

PROFILE_REPEATS = {"quick": 1, "standard": 3, "full": 5}
PROFILE_TESTS = {"quick": ["W1", "W2"], "standard": ["W1", "W2", "W3"], "full": ["W1", "W2", "W3"]}
MAX_STEPS = 12
MAX_FETCH_BYTES = 1_500_000
MAX_FETCH_TEXT = 24_000

SYSTEM = (
    "You are a research-capable technical assistant. For every task in this suite you MUST use the provided "
    "web_search and/or web_fetch tools rather than relying only on memory. Prefer primary/official sources, "
    "distinguish sourced facts from inference, and include the exact URLs supporting your answer. If sources "
    "conflict, say so. Do not invent citations."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the current public web through the local SearXNG metasearch service.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch readable text from one public HTTP/HTTPS webpage. Private/local addresses are blocked.",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
        },
    },
]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.title: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "svg"}:
            self.skip += 1
        if t == "title":
            self.in_title = True
        if t in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"} and not self.skip:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1
        if t == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        text = data.strip()
        if not text:
            return
        if self.in_title:
            self.title.append(text)
        self.parts.append(text + " ")

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _cache_root(run_dir: Path) -> Path:
    return run_dir.parents[2] / "cache" / "web"


def _prepared(run_dir: Path) -> dict[str, Any]:
    p = _cache_root(run_dir) / "prepared.json"
    if not p.exists():
        raise RuntimeError("Web preflight manifest is missing")
    return load_json(p)


def job_count(repo_root: Path, profile: str) -> int:
    return len(PROFILE_TESTS[profile]) * PROFILE_REPEATS[profile]


def _validate_public_url(url: str) -> str:
    try:
        u = parse.urlsplit(url)
    except Exception as exc:
        raise ValueError(f"invalid URL: {exc}") from exc
    if u.scheme not in {"http", "https"}:
        raise ValueError("only http/https URLs are allowed")
    if not u.hostname:
        raise ValueError("URL has no hostname")
    host = u.hostname.rstrip(".")
    try:
        infos = socket.getaddrinfo(host, u.port or (443 if u.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"hostname resolution failed: {exc}") from exc
    addresses = {i[4][0] for i in infos}
    if not addresses:
        raise ValueError("hostname resolved to no address")
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"blocked non-public destination: {ip}")
    return parse.urlunsplit(u)


class _SafeRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        safe = _validate_public_url(parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, safe)


def _search(searx_url: str, query: str, max_results: int) -> dict[str, Any]:
    max_results = max(1, min(int(max_results or 6), 10))
    params = parse.urlencode({"q": query, "format": "json", "language": "all", "safesearch": 0})
    url = searx_url.rstrip("/") + "/search?" + params
    req = request.Request(url, headers={"User-Agent": "Benchmark/1.0"})
    with request.urlopen(req, timeout=20) as resp:
        raw = resp.read(MAX_FETCH_BYTES)
    obj = json.loads(raw.decode("utf-8", errors="replace"))
    rows = []
    for item in (obj.get("results") or [])[:max_results]:
        rows.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content"),
                "engine": item.get("engine"),
                "engines": item.get("engines"),
                "publishedDate": item.get("publishedDate"),
            }
        )
    return {"query": query, "count": len(rows), "results": rows}


def _fetch(url: str) -> dict[str, Any]:
    safe = _validate_public_url(url)
    opener = request.build_opener(_SafeRedirect())
    req = request.Request(
        safe,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Benchmark/1.0; +https://example.invalid)",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5",
        },
    )
    with opener.open(req, timeout=20) as resp:
        final_url = _validate_public_url(resp.geturl())
        content_type = (resp.headers.get("Content-Type") or "").lower()
        raw = resp.read(MAX_FETCH_BYTES + 1)
        truncated = len(raw) > MAX_FETCH_BYTES
        raw = raw[:MAX_FETCH_BYTES]
        status = getattr(resp, "status", 200)
    charset = "utf-8"
    m = re.search(r"charset=([^;\s]+)", content_type)
    if m:
        charset = m.group(1).strip('"\'')
    text = raw.decode(charset, errors="replace")
    title = ""
    if "html" in content_type or "<html" in text[:500].lower():
        parser = _TextExtractor()
        parser.feed(text)
        readable = parser.text()
        title = " ".join(parser.title).strip()
    elif "json" in content_type:
        try:
            readable = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:
            readable = text
    else:
        readable = text
    readable = html.unescape(readable)
    readable = readable[:MAX_FETCH_TEXT]
    return {
        "requested_url": url,
        "url": final_url,
        "status": status,
        "content_type": content_type,
        "title": title,
        "text": readable,
        "download_truncated": truncated,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _chat(api: str, model: str, messages: list[dict[str, Any]], mode: str, temperature: float, seed: int, context: int):
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": context},
    }
    include, value = mode_to_think(mode)
    if include:
        payload["think"] = value
    req = request.Request(
        api.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    sampler = PowerSampler()
    sampler.start()
    started = time.monotonic()
    try:
        try:
            with request.urlopen(req, timeout=None) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    finally:
        elapsed = time.monotonic() - started
        power = sampler.stop(elapsed)
    return payload, obj, elapsed, power


def _urls(text: str) -> list[str]:
    found = re.findall(r"https?://[^\s<>\]\[)('\\\"`]+", text)
    return [u.rstrip(".,;:!?") for u in found]


def _norm_url(url: str) -> str:
    try:
        p = parse.urlsplit(url)
        return parse.urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/") or "/", p.query, ""))
    except Exception:
        return url.rstrip("/")


def run_web(
    *,
    repo_root: Path,
    run_dir: Path,
    model: str,
    mode: str,
    profile: str,
    api: str,
    temperature: float,
    seed: int,
    context: int,
    is_completed: Callable[[str], bool],
    mark_completed: Callable[[str, dict[str, Any]], None],
    should_stop: Callable[[], bool],
    set_current: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    prep = _prepared(run_dir)
    searx_url = prep["url"]
    tests_all = {x["id"]: x for x in load_json(repo_root / "benchmarks" / "web" / "tests.json")["tests"]}
    repeats = PROFILE_REPEATS[profile]
    config_slug = (model + "__" + mode).replace("/", "_").replace(":", "_")
    base = run_dir / "raw" / "web" / config_slug
    summaries: list[dict[str, Any]] = []

    for test_id in PROFILE_TESTS[profile]:
        test = tests_all[test_id]
        test_rows: list[dict[str, Any]] = []
        for repeat in range(1, repeats + 1):
            if should_stop():
                return {"stopped": True, "model": model, "mode": mode, "profile": profile, "tests": summaries}
            key = f"web|{model}|{mode}|{test_id}|{repeat}"
            out_dir = base / test_id / f"repeat-{repeat}"
            result_path = out_dir / "result.json"
            if is_completed(key) and result_path.exists():
                test_rows.append(load_json(result_path))
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": test["prompt"]},
            ]
            calls: list[dict[str, Any]] = []
            search_calls = 0
            fetch_calls = 0
            fetched_urls: list[str] = []
            search_urls: list[str] = []
            final_answer = ""
            unsupported = False

            for step in range(1, MAX_STEPS + 1):
                if should_stop():
                    return {"stopped": True, "model": model, "mode": mode, "profile": profile, "tests": summaries}
                set_current({
                    "benchmark": "web", "model": model, "mode": mode, "test": test_id,
                    "repeat": repeat, "repeats": repeats, "turn": step, "turns": MAX_STEPS,
                })
                try:
                    payload, resp, elapsed, power = _chat(api, model, messages, mode, temperature, seed, context)
                except RuntimeError as exc:
                    if "tool" in str(exc).lower() and ("support" in str(exc).lower() or "does not" in str(exc).lower()):
                        unsupported = True
                        calls.append({"step": step, "error": str(exc), "unsupported": True})
                        break
                    raise
                msg = resp.get("message") or {}
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                record: dict[str, Any] = {
                    "step": step,
                    "request": payload,
                    "response": resp,
                    "metrics": response_metrics(resp, elapsed),
                    "power": power,
                    "tools": [],
                }
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if msg.get("thinking"):
                    assistant_message["thinking"] = msg.get("thinking")
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                if not tool_calls:
                    final_answer = content
                    calls.append(record)
                    break

                for tc in tool_calls:
                    fn = (tc.get("function") or {})
                    name = fn.get("name")
                    args = fn.get("arguments") or {}
                    if not isinstance(args, dict):
                        args = {}
                    tool_result: dict[str, Any]
                    ok = True
                    try:
                        if name == "web_search":
                            search_calls += 1
                            tool_result = _search(searx_url, str(args.get("query", "")), int(args.get("max_results") or 6))
                            search_urls.extend([r.get("url") for r in tool_result.get("results", []) if r.get("url")])
                        elif name == "web_fetch":
                            fetch_calls += 1
                            tool_result = _fetch(str(args.get("url", "")))
                            if tool_result.get("url"):
                                fetched_urls.append(tool_result["url"])
                        else:
                            ok = False
                            tool_result = {"error": f"unsupported tool: {name}"}
                    except Exception as exc:
                        ok = False
                        tool_result = {"error": f"{type(exc).__name__}: {exc}"}
                    record["tools"].append({"name": name, "args": args, "ok": ok, "result": tool_result})
                    messages.append({"role": "tool", "tool_name": str(name), "content": json.dumps(tool_result, ensure_ascii=False)})
                calls.append(record)
            else:
                final_answer = final_answer or ((calls[-1].get("response", {}).get("message") or {}).get("content") or "" if calls else "")

            citations = _urls(final_answer)
            fetched_norm = {_norm_url(x) for x in fetched_urls}
            valid_citations = [u for u in citations if _norm_url(u) in fetched_norm]
            checks = evaluate_checks(final_answer, test.get("checks", []))
            checks.extend([
                {"name": "used web_search", "passed": (search_calls > 0) if test.get("require_search") else True},
                {"name": "used web_fetch", "passed": (fetch_calls > 0) if test.get("require_fetch") else True},
                {"name": "minimum citation count", "passed": len(citations) >= int(test.get("minimum_citations", 1)), "count": len(citations)},
                {"name": "at least one citation matches a fetched page", "passed": bool(valid_citations), "valid_citations": valid_citations},
            ])
            passed_count = sum(1 for x in checks if x.get("passed"))
            result = {
                "test": test_id,
                "title": test["title"],
                "repeat": repeat,
                "model": model,
                "mode": mode,
                "completed_at": utc_now(),
                "unsupported": unsupported,
                "pass": None if unsupported else passed_count == len(checks),
                "checks_passed": passed_count,
                "checks_total": len(checks),
                "checks": checks,
                "steps": len(calls),
                "web_search_calls": search_calls,
                "web_fetch_calls": fetch_calls,
                "search_result_urls": list(dict.fromkeys(search_urls)),
                "fetched_urls": list(dict.fromkeys(fetched_urls)),
                "citations": citations,
                "valid_fetched_citations": valid_citations,
                "final_answer": final_answer,
                "searxng": prep,
                "performance": call_performance_summary(calls),
            }
            atomic_json(out_dir / "calls.json", calls)
            atomic_json(result_path, result)
            mark_completed(key, result)
            test_rows.append(result)

        summaries.append({
            "test": test_id,
            "title": test["title"],
            "repeats": test_rows,
            "passes": sum(1 for x in test_rows if x.get("pass") is True),
            "unsupported": sum(1 for x in test_rows if x.get("unsupported")),
            "repeat_count": len(test_rows),
        })

    rows = [r for t in summaries for r in t["repeats"]]
    task_wall = [r.get("performance",{}).get("task_totals",{}).get("wall_seconds") for r in rows]
    task_energy = [r.get("performance",{}).get("task_totals",{}).get("estimated_gpu_energy_wh") for r in rows]
    return {
        "stopped": False,
        "model": model,
        "mode": mode,
        "profile": profile,
        "searxng": prep,
        "tests": summaries,
        "passed_runs": sum(1 for r in rows if r.get("pass") is True),
        "total_runs": len(rows),
        "unsupported_runs": sum(1 for r in rows if r.get("unsupported")),
        "performance": {"task_wall_seconds": distribution_summary(task_wall), "task_estimated_gpu_energy_wh": distribution_summary(task_energy)},
    }
