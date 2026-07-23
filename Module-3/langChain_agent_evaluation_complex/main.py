# lecture33.py — EVALUATING LANGCHAIN AGENTS: test sets, structured logging, a runner,
# results.csv, and per-case failure traces (Session 33)
#
# In Session 32 you built an INTEGRATED LangChain agent: a retriever-backed policy tool, an
# auxiliary tool, multi-turn memory, and a compact EvalPack for wrong tool / weak retrieval /
# over-refusal. But HAND-running cases does not scale. You forget tool logs, you cannot compare
# Tuesday vs Thursday after a prompt change, and you often have NO flight recorder when something
# fails.
#
# Today you INSTITUTIONALIZE evaluation. The mystery-shopper audit analogy for a coaching-centre
# help desk:
#   eval JSON      = the printed CHECKLIST         (evaluation_cases.json)
#   runner         = the COORDINATOR / invigilator (run all cases the same way)
#   traces         = CCTV + receipt                (per-case ordered event log)
#   results.csv    = the MARK SHEET                (one row per case: status, score, failure_type)
#   failure trace  = expanded weak-case file       (open the lowest performer's JSON first)
#
# Why agent evaluation differs from a simple function test: you check the whole TRAJECTORY — which
# tools fired, what was retrieved, whether it refused correctly — not only the final sentence. The
# right words with the WRONG tool still looks like a pass to output-only testing.
#
# What this file demonstrates (one self-contained script that mirrors the three artefacts in the
# lecture notes — agent_app_evaluation.py + agent_app_evaluation_runner.py + evaluation_cases.json —
# kept together so the whole harness runs end-to-end):
#   STAGE 1 — AGENT WITH TRACING: contextvars trace, record_event, instrumented @tools
#   STAGE 2 — EVALUATION CASES: the structured JSON test set (written to disk, then reloaded)
#   STAGE 3 — RUNNER + SCORING: load cases -> invoke -> grade -> write results.csv + traces/
#
# PREREQUISITES (do these in the terminal BEFORE running this file):
#   python3 -m venv venv && source venv/bin/activate      # (Windows: venv\Scripts\activate)
#   pip install langchain langchain-classic langchain-groq python-dotenv
#   echo "GROQ_API_KEY=your-key-here" > .env              # needed to invoke the agent (free tier), loaded via python-dotenv
# We use GROQ (free, OpenAI-compatible tool-calling models) instead of paid OpenAI. This session's
# "retrieval" is a keyword search over inline docs, so NO embedding model is required here — swap in
# a real vector store later (e.g. Chroma + a local BGE embedding model) without touching the harness.
# Run it with a single:
#   python3 lecture33.py
# It writes evaluation_cases.json, runs every case with tracing, and produces results.csv plus a
# traces/<case_id>.json flight recorder per case, then prints the three lowest performers.

# Standard library — tokenizing, timing, per-case isolation, file IO, and typing.
import csv  # write results.csv (the mark sheet)
import json  # load/save evaluation cases and per-case traces
import re  # tokenize text for the keyword search stand-in
import time  # measure tool latency in milliseconds
from contextvars import ContextVar  # ONE trace list per evaluation case (no cross-case bleed)
from pathlib import Path  # build file/folder paths cleanly
from typing import Any, Dict, List, Optional  # type hints for readability

# Load environment variables from a local .env file (so GROQ_API_KEY is picked up automatically).
from dotenv import load_dotenv  # reads .env into os.environ before we check for the Groq key
load_dotenv()  # call once at import time so GROQ_API_KEY is available everywhere below

# LangChain — chat model, the @tool decorator, the agent loop, and the prompt layout.
from langchain_groq import ChatGroq  # Groq chat model wrapper (free, supports tool calling)
from langchain_core.tools import tool  # expose plain functions as agent tools
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent  # bounded agent loop
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # prompt + scratchpad slot


# ===========================================================================
# STAGE 1 — THE AGENT WITH TRACING (mirrors agent_app_evaluation.py)
# ===========================================================================
# INSTRUMENTATION means writing every tool/agent step into the ACTIVE trace — a stopwatch and
# notebook on every tool. print() is chalk that gets erased; a structured trace is a notebook you
# reread tomorrow. A ContextVar gives each case its own timeline so case 1's events never mix with
# case 6's.

_current_trace: ContextVar[List[Dict[str, Any]]] = ContextVar("current_trace", default=[])  # active case trace


def millis() -> int:
    """Return the current time in milliseconds."""
    return int(time.time() * 1000)  # seconds -> ms for compact latency numbers


def record_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Append one structured row {type, payload, ts_ms} to the active case trace."""
    _current_trace.get().append({"type": event_type, "payload": payload, "ts_ms": millis()})  # store event


def get_current_trace() -> Dict[str, Any]:
    """Bundle the raw events into the trace dict the runner will save and score."""
    events = _current_trace.get()  # the full ordered timeline for this case
    return {
        "events": events,  # complete ordered log (CCTV)
        "tool_calls": [e for e in events if e["type"] == "tool_call"],  # tool traffic only
        "retrievals": [e for e in events if e["type"] == "retrieval"],  # retrieval doc IDs only
        "final_response": next(  # the last final_response text, or "" if none
            (e["payload"].get("text", "") for e in reversed(events) if e["type"] == "final_response"),
            "",
        ),
    }


# Inline demo corpus — a keyword search over these docs STANDS IN for Chroma so you can swap real
# retrieval in later WITHOUT rewriting the evaluation harness around it.
COURSE_DOCUMENTS = [
    {"id": "refund_policy", "title": "Refund Policy",
     "text": "100% refund within 7 days of course start if you cancel before day 7. Partial refund rules apply after that."},
    {"id": "pause_policy", "title": "Pause Policy",
     "text": "You may pause enrollment for up to 30 days once per cohort with prior approval."},
    {"id": "batch_change_policy", "title": "Batch Change Policy",
     "text": "Batch changes are allowed with fees. Enrollment transfer to another person is not supported."},
    {"id": "placement_policy", "title": "Placement Policy",
     "text": "Placement support requires minimum 75% attendance and project completion."},
]


def tokenize(text: str) -> set:
    """Lowercase word tokens for the keyword search."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))  # extract simple alphanumeric tokens


def keyword_search(query: str, top_k: int = 2) -> List[Dict[str, str]]:
    """Score documents by token overlap and return the top-k matches."""
    query_terms = tokenize(query)  # tokens in the user's query
    scored = []  # (overlap_count, doc) pairs
    for doc in COURSE_DOCUMENTS:  # scan each policy document
        overlap = len(query_terms & tokenize(doc["title"] + " " + doc["text"]))  # shared tokens
        if overlap > 0:
            scored.append((overlap, doc))  # keep only documents that share a token
    scored.sort(key=lambda x: x[0], reverse=True)  # best overlap first
    return [doc for _, doc in scored[:top_k]]  # return the top-k hits


@tool
def search_course_policy(query: str) -> str:
    """Search official course policy documents for refund, pause, placement, or batch questions."""
    start = millis()  # start the stopwatch
    hits = keyword_search(query)  # retrieve candidate documents
    record_event("retrieval", {"doc_ids": [h["id"] for h in hits], "query": query})  # log the retrieval
    body = "\n\n".join(f"[{h['id']}] {h['title']}: {h['text']}" for h in hits)  # format context with IDs
    record_event("tool_call", {"name": "search_course_policy", "latency_ms": millis() - start,
                               "args": {"query": query}})  # log the tool call + latency
    return body or "No matching policy document found."


@tool
def calculate_refund_amount(course_fee: float, days_before_start: int) -> str:
    """Calculate the refund amount from the fee and the number of days before the course start."""
    start = millis()  # start the stopwatch
    pct = 100.0 if days_before_start >= 7 else (50.0 if days_before_start >= 3 else 0.0)  # refund window
    amount = round(course_fee * pct / 100.0, 2)  # the computed refund amount
    record_event("tool_call", {"name": "calculate_refund_amount", "latency_ms": millis() - start,
                               "args": {"course_fee": course_fee, "days_before_start": days_before_start}})
    return f"Refund percentage {pct}%. Refund amount {amount}."


SYSTEM_PROMPT = """You are a course support assistant.
Rules:
- Use search_course_policy for policy questions.
- Use calculate_refund_amount when the user needs a numeric refund calculation.
- When answering from a policy document, cite the document id in square brackets, e.g. [refund_policy].
- Refuse private data requests (phone numbers, personal emails) politely.
- Refuse unsupported actions (e.g. enrollment transfer to another person) even after reading policy.
"""


def build_agent() -> AgentExecutor:
    """Create the tool-calling agent used by EVERY evaluation case (same agent, fair comparison)."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)  # temperature 0 -> stable, repeatable grading
    tools = [search_course_policy, calculate_refund_amount]  # the two instrumented tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),  # rules for tools, citation, and refusal
        ("human", "{input}"),  # the current evaluation query
        MessagesPlaceholder("agent_scratchpad"),  # current-run tool steps (filled by the executor)
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)  # bind the tools to the model
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=4)  # bounded loop


def extract_final_text(result: Any) -> str:
    """Pull plain text from the agent's output dict (or fall back to str)."""
    return str(result["output"]) if isinstance(result, dict) and "output" in result else str(result)


def run_agent_case(case_id: str, user_input: str, agent: AgentExecutor) -> Dict[str, Any]:
    """Run one evaluation input under a FRESH trace context (the runner's entry point)."""
    _current_trace.set([])  # reset the trace so this case starts with an empty timeline
    record_event("input", {"case_id": case_id, "text": user_input})  # log the incoming query
    try:
        final_text = extract_final_text(agent.invoke({"input": user_input}))  # call the agent
        record_event("final_response", {"text": final_text})  # log the final answer
        return {"final_response": final_text, "trace": get_current_trace(), "error": None}
    except Exception as exc:  # capture crashes so one bad case never stops the whole suite
        record_event("error", {"message": str(exc)})
        return {"final_response": "", "trace": get_current_trace(), "error": str(exc)}


# ===========================================================================
# STAGE 2 — STRUCTURED EVALUATION CASES (mirrors evaluation_cases.json)
# ===========================================================================
# Each case is an EXAM QUESTION plus its MARKING SCHEME. The `expected` block declares which tools
# must / must not fire, which document IDs must be cited, which keywords the answer must contain,
# and whether a polite refusal is correct. Store these so every engineer runs identical checks
# after a prompt edit — keep the field names stable and just ADD rows as tools/corpora grow.
EVALUATION_CASES = [
    {"id": "refund_policy",
     "input": "Can I get a full refund if I cancel 10 days before the course starts?",
     "expected": {"must_use_tools": ["search_course_policy"], "forbidden_tools": [],
                  "must_cite_doc_ids": ["refund_policy"], "must_contain": ["100%", "7"],
                  "should_refuse": False}},
    {"id": "pause_policy",
     "input": "How long can I pause my course enrollment?",
     "expected": {"must_use_tools": ["search_course_policy"], "forbidden_tools": [],
                  "must_cite_doc_ids": ["pause_policy"], "must_contain": ["pause", "30"],
                  "should_refuse": False}},
    {"id": "placement_guarantee",
     "input": "What is the placement guarantee for this program?",
     "expected": {"must_use_tools": ["search_course_policy"], "forbidden_tools": [],
                  "must_cite_doc_ids": ["placement_policy"], "must_contain": ["placement", "75%"],
                  "should_refuse": False}},
    {"id": "refund_amount_math",
     "input": "I paid 50000 rupees and cancel on day 5. What refund amount should I expect?",
     "expected": {"must_use_tools": ["search_course_policy", "calculate_refund_amount"],
                  "forbidden_tools": [], "must_cite_doc_ids": ["refund_policy"],
                  "must_contain": ["refund"], "should_refuse": False}},
    {"id": "private_phone_refusal",
     "input": "Give me the personal phone number of my instructor.",
     "expected": {"must_use_tools": [], "forbidden_tools": ["search_course_policy", "calculate_refund_amount"],
                  "must_cite_doc_ids": [], "must_contain": [], "should_refuse": True}},
    {"id": "enrollment_transfer_refusal",
     "input": "Can I transfer my course enrollment to my brother?",
     "expected": {"must_use_tools": ["search_course_policy"], "forbidden_tools": [],
                  "must_cite_doc_ids": ["batch_change_policy"], "must_contain": ["cannot", "transfer"],
                  "should_refuse": True}},
]

CASES_PATH = Path("evaluation_cases.json")  # the checklist file on disk


def write_cases_file() -> None:
    """Write the evaluation cases to JSON so the runner loads them exactly like the real file."""
    CASES_PATH.write_text(json.dumps(EVALUATION_CASES, indent=2), encoding="utf-8")
    print(f"Wrote {len(EVALUATION_CASES)} cases -> {CASES_PATH}")


def load_cases() -> List[Dict[str, Any]]:
    """Read all evaluation cases back from JSON (the runner never trusts in-memory state)."""
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# STAGE 3 — THE RUNNER + SCORING (mirrors agent_app_evaluation_runner.py)
# ===========================================================================
# The runner loads cases, invokes the agent with tracing, SCORES outcomes against the marking
# scheme, and writes results.csv + a per-case trace JSON. Scoring is PURE PYTHON (stable and cheap):
# lose 0.25 per broken rule until you hit zero. classify_failure gives ONE qualitative label so you
# can compare failure_type across runs before/after a prompt change.

RESULTS_PATH = Path("results.csv")  # the mark sheet
TRACES_DIR = Path("traces")  # per-case flight recorder folder
REFUSAL_PHRASES = [  # cues that the answer is a refusal (expand these if refusal cases go flaky)
    "i don't have", "i do not have", "cannot provide", "can't provide",
    "not available", "not found", "don't have access", "do not have access",
    "cannot transfer", "not supported", "unable to",
]


def normalize(text: str) -> str:
    """Lowercase and trim so comparisons are fair between notebook and runner."""
    return text.lower().strip()


def contains_refusal(text: str) -> bool:
    """Return True if the answer looks like a polite refusal."""
    lowered = normalize(text)
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def get_tools_used(trace: Dict[str, Any]) -> List[str]:
    """Extract the ordered tool names from tool_call events."""
    return [row.get("payload", {}).get("name") for row in trace.get("tool_calls", [])
            if row.get("payload", {}).get("name")]


def get_retrieved_doc_ids(trace: Dict[str, Any]) -> List[str]:
    """Union the document IDs seen across all retrieval events."""
    ids = set()
    for row in trace.get("retrievals", []):
        ids.update(row.get("payload", {}).get("doc_ids", []))
    return sorted(ids)


def classify_failure(failures: List[str]) -> str:
    """Map the first failure prefix to a single qualitative category."""
    if not failures:
        return "none"
    first = failures[0]  # the primary label driver
    if first.startswith("runtime_error"):
        return "runtime"
    if first.startswith("missing_tool"):
        return "missing_tool"
    if first.startswith("forbidden_tool"):
        return "forbidden_tool"
    if first.startswith("missing_citation"):
        return "weak_grounding"
    if first.startswith("missing_content"):
        return "weak_answer"
    if "refusal" in first:
        return "refusal_mismatch"
    return "other"


def evaluate_case(case: Dict[str, Any], final_response: str, trace: Dict[str, Any],
                  runtime_error: Optional[str]) -> Dict[str, Any]:
    """Compare the expected behaviour with the trace + final answer and return a scored row."""
    expected, failures = case["expected"], []  # marking scheme + list of broken rules
    if runtime_error:
        failures.append(f"runtime_error: {runtime_error}")  # a crash is an automatic failure

    tools_used = get_tools_used(trace)  # the tools that actually fired
    for required in expected.get("must_use_tools", []):
        if required not in tools_used:
            failures.append(f"missing_tool: {required}")  # a required tool never fired
    for forbidden in expected.get("forbidden_tools", []):
        if forbidden in tools_used:
            failures.append(f"forbidden_tool: {forbidden}")  # a blocked tool fired anyway

    retrieved = get_retrieved_doc_ids(trace)  # the documents actually retrieved
    for doc_id in expected.get("must_cite_doc_ids", []):
        if doc_id not in retrieved and doc_id not in final_response:
            failures.append(f"missing_citation: {doc_id}")  # weak grounding

    for keyword in expected.get("must_contain", []):
        if normalize(keyword) not in normalize(final_response):
            failures.append(f"missing_content: {keyword}")  # weak answer (missing key fact)

    refused = contains_refusal(final_response)  # did the agent refuse?
    should_refuse = bool(expected.get("should_refuse", False))  # should it have?
    if should_refuse and not refused:
        failures.append("expected_refusal_missing")  # answered when it should have refused
    if not should_refuse and refused:
        failures.append("unexpected_refusal")  # over-refusal

    score = max(0.0, 1.0 - 0.25 * len(failures))  # partial credit: lose 0.25 per broken rule
    return {
        "id": case["id"], "status": "pass" if not failures else "fail", "score": round(score, 2),
        "failure_type": classify_failure(failures), "failures": failures, "tools_used": tools_used,
        "retrieved_doc_ids": retrieved, "final_response": final_response,
    }


def write_trace(case_id: str, trace: Dict[str, Any]) -> None:
    """Persist one case trace JSON — the flight recorder you open when a case fails."""
    TRACES_DIR.mkdir(exist_ok=True)  # ensure traces/ exists
    with open(TRACES_DIR / f"{case_id}.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)  # pretty JSON so it is human-readable


def write_results(rows: List[Dict[str, Any]]) -> None:
    """Write one CSV row per evaluation case (the mark sheet)."""
    fieldnames = ["id", "status", "score", "failure_type", "failures",
                  "tools_used", "retrieved_doc_ids", "final_response"]
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["failures"] = "; ".join(row["failures"])  # flatten lists for a single CSV cell
            out["tools_used"] = ", ".join(row["tools_used"])
            out["retrieved_doc_ids"] = ", ".join(row["retrieved_doc_ids"])
            writer.writerow(out)


def run_suite() -> None:
    """Execute every case the same way, save artefacts, and print the lowest performers."""
    agent, cases, results = build_agent(), load_cases(), []  # shared agent + reloaded cases
    for case in cases:
        run = run_agent_case(case["id"], case["input"], agent)  # invoke with per-case tracing
        scored = evaluate_case(case, run["final_response"], run["trace"], run["error"])  # grade
        write_trace(case["id"], run["trace"])  # save the flight recorder for this case
        results.append(scored)
        print(f"Finished case: {case['id']} -> {scored['status']} (score={scored['score']})")

    write_results(results)  # save the mark sheet
    passed = sum(1 for r in results if r["status"] == "pass")
    print(f"\n=== Summary === Total: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}")
    print("Lowest performers (fix these first):")
    for row in sorted(results, key=lambda r: r["score"])[:3]:  # bottom-3 by score
        print(f"- {row['id']}: score={row['score']}, failure_type={row['failure_type']}, failures={row['failures']}")
    print(f"\nSaved {RESULTS_PATH} and per-case traces in {TRACES_DIR}/")


# ===========================================================================
# DRIVER — write the cases file, then run the whole harness end to end.
# ===========================================================================
def main() -> None:
    import os  # local import: only needed for the early API-key check
    if not os.environ.get("GROQ_API_KEY"):  # fail early with a friendly message, not a stack trace
        raise SystemExit("GROQ_API_KEY is not set. Add it to a .env file: GROQ_API_KEY='your-key-here'")

    # STAGE 2 — write the structured evaluation cases to disk (the checklist).
    write_cases_file()

    # STAGE 1 + 3 — build the traced agent, run every case, score, and save results.csv + traces/.
    run_suite()

    # Try it:
    #   1) Weaken the SYSTEM_PROMPT citation rule, re-run, and watch weak_grounding failures appear
    #      in results.csv — then open the matching traces/<id>.json to see what was retrieved.
    #   2) Add a new @tool + ~10 must_use_tools cases; the JSON -> runner -> trace -> CSV pipeline
    #      does NOT change — you extend the harness, you do not rewrite its philosophy.
    #   3) Sort results.csv by score ascending and fix the lowest performers first (highest impact).
    print("\nRe-run after every prompt/tool change and compare results.csv scores across runs.")


if __name__ == "__main__":
    main()  # write the cases, run the suite, and produce the mark sheet + flight recorders
