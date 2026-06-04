from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph

from code_assist.rag.retriever import query_repo
from code_assist.tools.files import read_file, search_code

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "where",
    "which",
    "you",
}


class AgentState(TypedDict):
    question: str
    memory: str
    retrieved: str
    tool_output: str
    action: str
    action_input: str
    candidate_paths: list[str]
    observations: list[str]
    steps: int
    answer: str


@dataclass(frozen=True)
class AgentConfig:
    chat_model: str
    embed_model: str
    root: Path


def build_graph(config: AgentConfig) -> StateGraph:
    graph = StateGraph(AgentState)

    def _truncate_query(text: str, limit: int = 512) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit]

    def _keyword_query(question: str) -> str:
        tokens = [
            token.strip(".,:;!?()[]{}\"'`/")
            for token in question.lower().split()
        ]
        keywords = [token for token in tokens if token and token not in STOPWORDS]
        return " ".join(keywords[:6])


    def retrieve(state: AgentState) -> AgentState:
        query = _truncate_query(state["question"])
        results = query_repo(
            query,
            config.root / ".ai-copilot" / "vectorstore",
            embeddings_model=config.embed_model,
        )
        if not results:
            return {"retrieved": ""}
        formatted = []
        for item in results:
            formatted.append(
                f"SOURCE: {item['source']}\nSCORE: {item['score']}\n{item['content']}"
            )
        return {"retrieved": "\n\n".join(formatted)}

    def plan(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a local AI dev copilot. Decide whether to use a tool. "
                    "Available tools:\n"
                    "- search_code(query): find matching lines\n"
                    "- read_file(path): read a specific file\n"
                    "If evidence is missing, you MUST choose a tool.\n"
                    "If you can answer now, choose action 'none'.\n"
                    "Respond ONLY as JSON: {{\"action\": \"search|read|none\", \"input\": \"...\"}}",
                ),
                (
                    "user",
                    "Memory:\n{memory}\n\nQuestion: {question}\n\nRetrieved Context:\n{retrieved}\n\nObservations:\n{observations}",
                ),
            ]
        )
        llm = ChatOllama(model=config.chat_model)
        message = prompt.format_messages(
            question=state["question"],
            memory=state.get("memory", ""),
            retrieved=state.get("retrieved", ""),
            observations="\n".join(state.get("observations", [])),
        )
        response = llm.invoke(message)
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError:
            return {"action": "none", "action_input": ""}

        action = str(payload.get("action", "none")).lower()
        action_input = str(payload.get("input", "")).strip()
        if action not in {"search", "read", "none"}:
            action = "none"
        if action == "none" and not state.get("observations"):
            action = "search"
            action_input = _keyword_query(state["question"]) or state["question"]
        if action == "search" and action_input and state.get("observations"):
            if state["observations"][-1].strip() == "No matches":
                action_input = _keyword_query(state["question"]) or action_input
        return {"action": action, "action_input": action_input}

    def run_tools(state: AgentState) -> AgentState:
        action = state.get("action", "none")
        action_input = state.get("action_input", "")
        output = ""
        candidate_paths: list[str] = list(state.get("candidate_paths", []))
        if action == "search" and action_input:
            matches = search_code(action_input, config.root)
            candidate_paths = []
            for match in matches:
                path = match.split(":", 1)[0]
                if path.endswith(".py") and path not in candidate_paths:
                    candidate_paths.append(path)
            output = "\n".join(dict.fromkeys(matches)) if matches else "No matches"
        elif action == "read" and action_input:
            output = read_file(action_input, config.root)
        elif action != "none":
            output = "No tool output"

        observations = list(state.get("observations", []))
        if output:
            observations.append(output)
        return {
            "tool_output": output,
            "observations": observations,
            "candidate_paths": candidate_paths,
            "steps": state.get("steps", 0) + 1,
        }

    def answer(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a local AI dev copilot. Use the provided context and tool output. "
                    "Be concise and practical. Format the response as:\n"
                    "Answer: <2-4 sentences>\n"
                    "Evidence: bullet list of file paths or snippets used (must appear in Retrieved Context or Observations)\n"
                    "Next: 0-2 action steps if needed\n"
                    "If evidence is missing, say 'Insufficient evidence' and suggest the next tool to run.",
                ),
                (
                    "user",
                    "Memory:\n{memory}\n\nQuestion: {question}\n\nRetrieved Context:\n{retrieved}\n\nObservations:\n{observations}",
                ),
            ]
        )
        llm = ChatOllama(model=config.chat_model)
        message = prompt.format_messages(
            question=state["question"],
            memory=state.get("memory", ""),
            retrieved=state.get("retrieved", ""),
            observations="\n".join(state.get("observations", [])),
        )
        response = llm.invoke(message)
        return {"answer": response.content}

    def should_continue(state: AgentState) -> str:
        if state.get("action") == "none":
            return "answer"
        if state.get("steps", 0) >= 2:
            return "answer"
        if state.get("action") == "search":
            candidates = state.get("candidate_paths", [])
            if candidates:
                state["action"] = "read"
                state["action_input"] = candidates[0]
                return "tools"
        return "tools"

    graph.add_node("retrieve", retrieve)
    graph.add_node("plan", plan)
    graph.add_node("tools", run_tools)
    graph.add_node("answer", answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_conditional_edges("plan", should_continue, {
        "tools": "tools",
        "answer": "answer",
    })
    graph.add_edge("tools", "plan")
    graph.set_finish_point("answer")

    return graph
