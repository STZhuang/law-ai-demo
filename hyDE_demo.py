import os
import logging
import sys
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.agent.react.types import ObservationReasoningStep, ActionReasoningStep, ResponseReasoningStep
from llama_index.core.agent import Task, AgentChatResponse
from llama_index.core.query_pipeline import AgentInputComponent, AgentFnComponent, QueryPipeline as QP, \
    ToolRunnerComponent
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.tools import BaseTool
from llama_index.core.llms import ChatResponse
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent import QueryPipelineAgentWorker
from llama_index.core.callbacks import CallbackManager
from typing import Dict, Any, List

# 假设这个函数在 init_model.py 中定义
from init_model import get_model, get_embedding

# 设置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 初始化模型
llm_model = get_model()
embedding_model = get_embedding()

# 设置全局设置
Settings.llm = llm_model
Settings.embed_model = embedding_model

# 初始化向量存储
db = chromadb.PersistentClient(".\\LawDb")
law_db = db.get_collection("law")
law_collection = ChromaVectorStore(chroma_collection=law_db)
law_index = VectorStoreIndex.from_vector_store(vector_store=law_collection, embed_model=embedding_model)

# 创建查询引擎
law_engine = law_index.as_query_engine(similarity_top_k=3)
hyde = HyDEQueryTransform(include_original=True)
law_hyde_engine = TransformQueryEngine(law_engine, hyde)

# 定义查询引擎工具
query_engine_tools = [
    QueryEngineTool(
        query_engine=law_hyde_engine,
        metadata=ToolMetadata(
            name="law",
            description="用户询问关于法律的任何信息时，调用此工具。"
        )
    ),
]


# 定义 agent 相关函数
def agent_input_fn(task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
    if "current_reasoning" not in state:
        state["current_reasoning"] = []
    reasoning_step = ObservationReasoningStep(observation=task.input)
    state["current_reasoning"].append(reasoning_step)
    return {"input": task.input}


def react_prompt_fn(task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]) -> List[ChatMessage]:
    chat_formatter = ReActChatFormatter()
    return chat_formatter.format(
        tools,
        chat_history=task.memory.get() + state["memory"].get_all(),
        current_reasoning=state["current_reasoning"],
    )


def parse_react_output_fn(task: Task, state: Dict[str, Any], chat_response: ChatResponse):
    from llama_index.core.agent.react.output_parser import ReActOutputParser
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}


def run_tool_fn(task: Task, state: Dict[str, Any], reasoning_step: ActionReasoningStep):
    tool_runner_component = ToolRunnerComponent(query_engine_tools, callback_manager=task.callback_manager)
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    observation_step = ObservationReasoningStep(observation=str(tool_output["output"]))
    state["current_reasoning"].append(observation_step)
    return {"response_str": observation_step.get_content(), "is_done": False}


def process_response_fn(task: Task, state: Dict[str, Any], response_step: ResponseReasoningStep):
    state["current_reasoning"].append(response_step)
    response_str = response_step.response
    state["memory"].put(ChatMessage(content=task.input, role=MessageRole.USER))
    state["memory"].put(ChatMessage(content=response_str, role=MessageRole.ASSISTANT))
    return {"response_str": response_str, "is_done": True}


def process_agent_response_fn(task: Task, state: Dict[str, Any], response_dict: dict):
    return AgentChatResponse(response_dict["response_str"]), response_dict["is_done"]


# 创建查询管道
qp = QP(verbose=True)

qp.add_modules({
    "agent_input": AgentInputComponent(fn=agent_input_fn),
    "react_prompt": AgentFnComponent(fn=react_prompt_fn),
    "llm": llm_model,
    "react_output_parser": AgentFnComponent(fn=parse_react_output_fn),
    "run_tool": AgentFnComponent(fn=run_tool_fn),
    "process_response": AgentFnComponent(fn=process_response_fn),
    "process_agent_response": AgentFnComponent(fn=process_agent_response_fn),
})

qp.add_chain(["agent_input", "react_prompt", "llm", "react_output_parser"])

qp.add_link(
    "react_output_parser",
    "run_tool",
    condition_fn=lambda x: not x["done"],
    input_fn=lambda x: x["reasoning_step"],
)

qp.add_link(
    "react_output_parser",
    "process_response",
    condition_fn=lambda x: x["done"],
    input_fn=lambda x: x["reasoning_step"],
)

qp.add_link("process_response", "process_agent_response")
qp.add_link("run_tool", "process_agent_response")

# 创建 agent
agent_worker = QueryPipelineAgentWorker(qp)
agent = agent_worker.as_agent(callback_manager=CallbackManager([]), verbose=True)

# 使用 agent
task = agent.create_task("杀人犯什么法律？")
step_output = agent.run_step(task.task_id)

print(step_output)
