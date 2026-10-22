Agent model: `claude-opus-5-5` (effort medium), judge: `claude-opus-5-5`, gold snapshot 2026-10-21.

| slice | n | tool_selection | first_call | call_precision | facts | judge_correct | judge_correct_or_partial | faithfulness |
|---|---|---|---|---|---|---|---|---|
| **all** | 40 | 97.5% | 100.0% | 100.0% | 100.0% | 95.0% | 100.0% | 0.905 |
| embed | 3 | 66.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.622 |
| emdb | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.974 |
| geo | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 77.8% | 100.0% | 0.925 |
| multi | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.923 |
| retrieval | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.914 |
| uniprot | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.897 |

Mean tool calls/question: 1.50; median latency: 8.8s
