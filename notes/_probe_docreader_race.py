"""Probe: does EVISEARCH_STAGE_CONCURRENCY on the arbiter make document_reader._documents unsafe?

No model, no GPU: pdf_query.build_document_input is replaced with a slow fake.
"""
import sys, time, threading, types
sys.path.insert(0, ".")
from src.evisearch.services import document_reader as dr


class FakeSpec:
    context_tokens = 131072
    image_tokens = None


class FakeChat:
    key = "qwen3.6-27b"
    spec = FakeSpec()

    class capabilities:
        images = True


dr.SELECTION = types.SimpleNamespace(option=lambda n: "pdf")
builds = {"n": 0}


def build(*a, **k):
    builds["n"] += 1
    time.sleep(0.25)
    return "DOC"


dr.pdf_query = types.SimpleNamespace(
    document_token_budget=lambda *a, **k: 100000,
    IMAGE_RULES="",
    build_document_input=build,
)

chat = FakeChat()

# --- part 1: 8 concurrent batches, as EVISEARCH_STAGE_CONCURRENCY=8 would give the arbiter
errors, ok = [], []


def call():
    try:
        ok.append(dr.document_for(chat, "doc-1"))
    except BaseException as e:
        errors.append(f"{type(e).__name__}: {e}")


ts = [threading.Thread(target=call) for _ in range(8)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(f"part 1: returned ok={len(ok)} errors={errors} redundant_builds={builds['n']} (1 is the serial count)")

# --- part 2: force the check->clear->set window that loses the key between set and return
dr._documents.clear()
gate = threading.Event()
builds["n"] = 0


def build_first_slow(*a, **k):
    builds["n"] += 1
    if builds["n"] == 1:
        gate.set()
        time.sleep(0.2)  # thread A is inside build; it will assign the key next
    return "DOC"


dr.pdf_query.build_document_input = build_first_slow


def late_clear():
    gate.wait()
    time.sleep(0.25)  # wake just after A's assignment, before A's return
    dr._documents.clear()  # this is line 50 of document_for, reached late by another batch


threading.Thread(target=late_clear, daemon=True).start()
errors2 = []


def call2():
    try:
        dr.document_for(chat, "doc-1")
    except BaseException as e:
        errors2.append(f"{type(e).__name__}: {e!r}")


t = threading.Thread(target=call2)
t.start()
t.join()
time.sleep(0.6)
print("part 2 (forced interleaving):", errors2 or "no error on this interleaving")
