# Minimalistic Multiagent Framework

This is part of the **Agentic Patterns Series** from [*The Neural Maze*](https://theneuralmaze.substack.com), which is a series of notes about understanding agents from scratch:

0. [What is an Agent?](https://theneuralmaze.substack.com/p/what-is-an-agent)

1. [Reflection Pattern: When Agents think twice](https://theneuralmaze.substack.com/p/reflection-pattern-agents-that-think)
    - [Andrew Ng Notes](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io)

The framework here is inspired by two fundamental concepts from CrewAI: **Crew** and **Agent**.

- **Crew**: A Crew is a collaborative group of agents working together to accomplish a set of tasks.  
    *Example:* In a document processing system, a Crew could consist of agents for text extraction, summarization, and translation, all working together to process incoming documents.

- **Agent**: An Agent is an autonomous unit responsible for executing specific tasks.  
    *Example:* The summarization agent focuses solely on condensing text, while the translation agent handles converting text from one language to another.

This framework allows you to define Crews and Agents to solve complex problems by breaking them down into manageable, specialized tasks.

<!-- ![CrewAI fundamental concepts](./images/crewai-fundamental-2-topics.png) -->