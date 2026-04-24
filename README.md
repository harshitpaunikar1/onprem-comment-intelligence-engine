# On-Prem Comment Intelligence Engine

> **Domain:** Marketing / AdTech

## Overview

Customer-facing teams drown in constant comment, update, and thread streams. Manually skimming everything wastes hours, slows response times, risks missing urgent issues. Leaders need trustworthy answers drawn from the latest internal data without exposing anything to third parties. The business needed a private, always-on assistant summarising conversations, answering questions from current databases, and flagging critical information the moment it appears. Without it, teams face higher handling costs, delayed resolutions, avoidable churn, and privacy exposure if data leaves organisations. The objective: compress review time, surface what truly matters, and keep processing fully within company-controlled infrastructure.

## Approach

- Mapped comment sources, triage rules, escalation paths with stakeholders; defined critical signals and SLAs
- Connected to internal database; indexed updates for retrieval so answers reflect most recent records
- Hosted Gemma-31B locally; built RAG pipeline to summarise threads and answer questions without external data egress
- Implemented message-queue workflow to prioritise requests, stream responses, route negotiation-stage items to humans
- Added proactive alerts: when critical details appear, prompt users to clarify/emphasise, trigger follow-up actions; full audit trail
- Containerised services with Docker and deployed on Heroku for repeatable releases; validated with golden sets, precision/recall checks, latency budgets

## Skills & Technologies

- Local LLM Hosting
- Retrieval-Augmented Generation
- Text Summarisation
- Intent Classification
- Message Queue Architecture
- Docker Containerization
- Heroku Deployment
- Data Privacy Controls
