Part of [ARCHITECTURE.md](../../ARCHITECTURE.md); the index there lists every part.

## 4. Error handling

Failure is expected and handled at three levels:

1. **Per topic.** `call_and_save_prompt` catches, logs and continues. One bad topic never
   ends a run.
2. **Per response.** On a validation error `operations._support.persist` writes the raw
   text to `{filename}_ERROR` and writes **nothing** at the atom's own `.json` path — so
   a failed regeneration can never overwrite a good atom with a placeholder (the
   `EmptyModel`-over-the-atom behaviour was retired; see B11 below).
3. **Per pass.** The `fix` pass is the deliberate second chance at every `_ERROR` file
   produced earlier in the run.

`_ERROR` files are therefore a work queue, not a dead end.

**Nothing is decided by a check that a concurrent writer can invalidate**
(B12, closed). `persist` originally asked `path.exists()` and then
wrote — two operations with a gap. Two calls for one topic resolve to one
content-addressed path, and the create pass can genuinely see a topic twice (two
`History` files may list the same city), so with concurrency above 1 a failing
call could see nothing there, a succeeding call could write a valid atom, and the
failing one could then replace it. The placeholder write is now `store.write_new`
— `O_CREAT | O_EXCL`, so the decision and the write are one atomic step — and
`persist` additionally takes a per-path lock, which orders a success against a
failure within the process. The lock covers this process; `O_EXCL` covers a
second process running the same phase.

**A failed *create* never destroys what it was regenerating** (B11,
closed). `LLMCall.write_response_to_file` writes whatever
`parse_response_to_pydantic` produced, which is an `EmptyModel` sentinel when
validation failed — so it wrote the sentinel to the atom's own path. That was
survivable only while ids were a per-process counter, because a failed rerun
landed on a *different* filename; content-addressed ids made the path a function
of the topic, so the same failure lands on the existing file. The create and
append operations now go through `operations._support.persist`, which writes a
validated atom, never overwrites an existing one with a sentinel, and leaves the
`_ERROR` file carrying the evidence. A *successful* regeneration retires a stale
sentinel — safe only because a validated atom was just written, the same
ordering as `commit_fix`.

**A failed augment never destroys what it was meant to improve** (B8,
closed). The fix and verify passes both call `get_valid_response()`, which
answers with an `EmptyModel` sentinel when the payload does not validate. The
old code wrote that answer straight to disk: a failed fix wrote a stub over the
target *and* deleted the `_ERROR` holding the only copy of what the model
returned, and a failed verify wrote a stub over a file that had been valid.
`execute_prompt._apply_fix` and `_apply_verify` now check for the sentinel before
writing anything, and `store.commit_fix` promotes the corrected file before
retiring the `_ERROR`, in that order.

**Truncation is reported as truncation.** A response that exhausts its output
budget arrives as incomplete JSON, so the naive report is `Invalid JSON: EOF
while parsing`, which blames the schema for what is really a `max_tokens`
problem. `Provider._validate` takes the provider's own stop signal (`length` on
OpenAI, `max_tokens` on Anthropic, `MAX_TOKENS` on Google) and, when parsing
fails after one of those, substitutes a message naming the model, the limit and
the tokens actually emitted. Reasoning models make this essential rather than
cosmetic: they spend output tokens on reasoning before emitting any content, so
an under-budgeted call returns an empty string and a validation error that
mentions nothing about tokens.

The batch path adds a fourth level, because a batch can lose results rather than
fail them. `providers/batch.py` reconciles every returned `custom_id` against the
submitted set, writes any shortfall to `data/automated/<domain>/_batches/missing/`, and
reports it on `CollectReport.missing`. Provider-side, API-level rejections (which OpenAI writes
to a separate error file) and job-level failures are both turned into per-request
failures rather than silence.
