# GPT
## [ReAct model](https://react-lm.github.io/)
```mermaid
graph TD
    label>ReAct]
    A[User input] --> B[System: Enhanced prompt];
    B --> C[System: Ingesting chat context, AI history, metadata];
    C --> D[LLM];
    D -->|Reasoning Traces| D
    D --> E[Thought]
    E --> F[Actions]
    F --> G{Observation}
    G -->|Done| H[Response]
    G -->|Rethink| D

```