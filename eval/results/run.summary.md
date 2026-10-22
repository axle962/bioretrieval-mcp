Agent model: `claude-opus-5-5` (effort medium), judge: `claude-opus-5-5`, gold snapshot 2026-10-21.

| slice | n | tool_selection | first_call | call_precision | facts | judge_correct | judge_correct_or_partial | faithfulness |
|---|---|---|---|---|---|---|---|---|
| **all** | 40 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.912 |
| embed | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.832 |
| emdb | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.949 |
| geo | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.923 |
| multi | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.944 |
| retrieval | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.929 |
| uniprot | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.873 |

Grading calls by model (server-side fallback on safety declines): `claude-opus-5` 1, `claude-opus-5-5` 119

Mean tool calls/question: 1.50; median latency: 8.8s
