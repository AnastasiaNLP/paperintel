# MCP Prompt Examples

These examples assume the PaperIntel MCP server is configured in your MCP client.
See [docs/MCP_SETUP.md](../docs/MCP_SETUP.md).

## Analyze One Paper

```text
Create a PaperIntel session as engineer and analyze https://arxiv.org/abs/1706.03762.
```

Expected tool flow:

1. `create_session`
2. `analyze_paper`

Analysis is synchronous and may take about one minute.

## Ask a Grounded Question

```text
Using the same PaperIntel session, what is the main contribution of the paper?
Include citations from the paper.
```

Expected tool:

1. `ask_paper`

## Inspect Session State

```text
Which papers are active in my current PaperIntel session?
```

Expected tool:

1. `get_session`

## Persona Examples

Engineer:

```text
Create a PaperIntel session as engineer and analyze https://arxiv.org/abs/1706.03762.
After it finishes, explain what I would need to implement from this paper.
```

Researcher:

```text
Create a PaperIntel session as researcher and analyze https://arxiv.org/abs/1706.03762.
After it finishes, explain the methodological contribution and limitations.
```

Tech lead:

```text
Create a PaperIntel session as techlead and analyze https://arxiv.org/abs/1706.03762.
After it finishes, summarize the engineering value, risk, and adoption cost.
```
