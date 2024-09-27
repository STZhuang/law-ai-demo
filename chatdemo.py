import logging
from pprint import pprint
from llama_index.llms.deepseekai import DeepSeek
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from init_model import get_embedding
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.agent import FunctionCallingAgentWorker, StructuredPlannerAgent
from config import Config
from load_from_chroma import get_index_from_chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = DeepSeek(temperature=1.3, **Config.get_openai_like_config())

resp = model.complete("""你好""")

pprint(resp.raw)

pprint(resp.raw.choices[0].message)

law_index = get_index_from_chroma(path=""".\\LawDb""", collection="""law""")

embedding_model = get_embedding()

Settings.embed_model = embedding_model

l = law_index.as_query_engine(llm=model)

l.query("""故意杀人判几年""")

tool = QueryEngineTool(
    query_engine=law_index.as_query_engine(llm=model),
    metadata=ToolMetadata(
        name="""law_retriever""",
        description="""
        当用户询问的是法律相关的问题时被调用""",
    ),
)

agent = FunctionCallingAgent.from_tools(
    [tool],
    llm=model,
    verbose=True,
)

import json

l = json.loads("""{"input": "\u6740\u4eba\u72af\u4ec0\u4e48\u6cd5\uff1f"}""")


res = agent.chat("""杀人犯什么法？""")

pprint(res)


llm.completion_to_prompt("""你哈珀""")

agent = ReActAgent.from_tools(
    tools=[tool],
    llm=llm,
    verbose=True,
)

agent.chat("""今天天气怎么样？""")

worker_agent = FunctionCallingAgentWorker.from_tools(
    tools=[tool],
    llm_model=model,
)

agent = StructuredPlannerAgent(worker_agent)
