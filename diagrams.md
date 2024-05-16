# On-Prem Comment Intelligence Engine Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## On-prem RAG architecture

```mermaid
flowchart TD
    N1["Step 1\nMapped comment sources, triage rules, escalation paths with stakeholders; defined "]
    N2["Step 2\nConnected to internal database; indexed updates for retrieval so answers reflect m"]
    N1 --> N2
    N3["Step 3\nHosted Gemma-31B locally; built RAG pipeline to summarise threads and answer quest"]
    N2 --> N3
    N4["Step 4\nImplemented message-queue workflow to prioritise requests, stream responses, route"]
    N3 --> N4
    N5["Step 5\nAdded proactive alerts: when critical details appear, prompt users to clarify/emph"]
    N4 --> N5
```

## Message queue processing flow

```mermaid
flowchart LR
    N1["Inputs\nLive yard-state entities such as docks, trailers, queues, and jockey availability"]
    N2["Decision Layer\nMessage queue processing flow"]
    N1 --> N2
    N3["User Surface\nOperator-facing UI or dashboard surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nSLA adherence"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
