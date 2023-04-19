from langchain.llms import OpenAI
from langchain.memory import DynamoDBChatMessageHistory

# from langchain.chains import ConversationChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.vectorstores import Chroma
from chromadb.config import Settings
import os
from langchain.embeddings import OpenAIEmbeddings
from uuid import uuid4
from typing import Dict, List, Any

from pydantic import BaseModel, Extra, root_validator
from langchain.prompts import PromptTemplate

from langchain.chains.conversation.prompt import PROMPT
from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import SequentialChain
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain


from fastapi import APIRouter

router = APIRouter()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
team_document = Chroma(
    collection_name="team_document",
    embedding_function=embeddings,
    client_settings=Settings(
        chroma_api_impl=os.getenv("CHROMA_API_IMPL"),
        chroma_server_host=os.getenv("CHROMA_SERVER_HOST"),
        chroma_server_http_port=8000,
    ),
)
team_conversation = Chroma(
    collection_name="team_conversation",
    embedding_function=embeddings,
    client_settings=Settings(
        chroma_api_impl=os.getenv("CHROMA_API_IMPL"),
        chroma_server_host=os.getenv("CHROMA_SERVER_HOST"),
        chroma_server_http_port=8000,
    ),
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
memory＿123 = ConversationBufferMemory()
memory＿234 = ConversationBufferMemory(memory_key="chat_history")

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]


@router.get("/")
async def ask(text: str) -> Any:
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0.9
    )  # 大的 temperature 会让输出有更多的随机性
    return llm(text)


class TeamDocument(BaseModel):
    text: str
    source: str


class TeamDocument1(BaseModel):
    text: str
    source: str = "QQ"


@router.post("/team_document")
async def add_chroma_documents(team_knowledge: TeamDocument) -> Any:
    document_id = str(uuid4())
    team_document.add_texts(
        texts=[team_knowledge.text],
        metadatas=[{"source": team_knowledge.source}],
        ids=[document_id],
    )
    return "ok"


@router.post("/team_conversation")
async def team_conversation(team_document: TeamDocument1) -> Any:
    document_id = str(uuid4())
    team_conversation.add_texts(
        texts=[team_document.text[:3000]],
        metadatas=[{"source": team_document.source}],
        ids=[document_id],
    )
    return "ok"


@router.get("/with-history")
async def ask_with_history(text: str) -> Any:
    chat_history = DynamoDBChatMessageHistory(
        table_name="conrad-conversation", session_id="123"
    )
    memory_567 = ConversationBufferMemory(chat_memory=chat_history)
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0.9
    )  # 大的 temperature 会让输出有更多的随机性
    conversation = ConversationChain(llm=llm, verbose=True, memory=memory_567)

    ai_response = conversation.predict(input=text)
    return ai_response


@router.get("/with-history-and-google-search")
async def ask_with_history_and_google_search(text: str) -> Any:
    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    memory＿234.chat_memory.add_user_message(text)
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory＿234
    )
    ai_response = agent_chain.run(input=text)
    memory＿234.chat_memory.add_ai_message(text)
    return ai_response


class HistoryAndEmbedding(BaseModel):
    question: str
    context: str


@router.post("/with-history-and-embedding")
async def ask_with_embedding(history_embedding: HistoryAndEmbedding) -> Any:
    question = history_embedding.question
    context = history_embedding.context
    document_id = str(uuid4())
    chain = load_summarize_chain(llm=OpenAI(temperature=0), chain_type="map_reduce")
    docs = [Document(page_content=question + context)]
    ans = chain.run(docs)

    team_document_docs = team_document.similarity_search(ans, k=4)
    team_conversation_docs = team_conversation.similarity_search(ans, k=20)
    context = team_document_docs
    question_prompt_template = """
    Based on {context} and {ans}, Give me a suggested response.
    {context}
    Question: {question}
    """
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template,
        input_variables=["context", "question", "ans"],
    )
    combine_prompt_template = """
    Given the following extracted parts of a long document and a question,
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    SOURCES:

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:
    """
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )
    chain = load_qa_chain(
        OpenAI(temperature=0),
        chain_type="map_reduce",
        return_map_steps=True,
        question_prompt=QUESTION_PROMPT,
        combine_prompt=COMBINE_PROMPT,
    )
    result = chain(
        {"input_documents": context, "question": question, "ans": ans},
        return_only_outputs=True,
    )

    team_conversation.add_texts(
        texts=["Human: " + ans + ",AI: " + result["output_text"]],
        metadatas=[{"source": "chat_with_ai"}],
        ids=[document_id],
    )
    team_prompt_template = """

        based on {question}, {ans}
        provide the teammate responses from {team_conversation} that answers {ans} with the teammate name.
        If there is no teammate response relating to {ans}, just return boolean False. Don't try to make up an answer.
    """
    TEAM_PROMPT = PromptTemplate(
        template=team_prompt_template,
        input_variables=["question", "ans", "team_conversation"],
    )
    chain1 = LLMChain(llm=OpenAI(temperature=0.9), prompt=TEAM_PROMPT)
    result11 = chain1.run(
        {"question": question, "ans": ans, "team_conversation": team_conversation_docs}
    )
    # return result
    if "False" == (result11.replace("\n", "").lstrip()):
        return result["output_text"]
    else:
        return (
            "Suggestion:"
            + result["output_text"]
            + "\nHere’s how other team members have responded to this question:"
            + result11
        )


@router.get("/summarization")
async def summarization(text: str) -> Any:
    chain = load_summarize_chain(llm=OpenAI(temperature=0), chain_type="map_reduce")
    docs = [Document(page_content=text)]
    ans = chain.run(docs)
    return ans


@router.get("/team_page")
async def tema_page(
    question: str,
) -> Any:
    team_document_docs = team_document.similarity_search(question, k=30)
    llm1 = OpenAI(temperature=0.7)
    prompt_template1 = PromptTemplate(
        input_variables=["question", "team_document_docs"],
        template="""
            Based on {team_document_docs}, tell me the if the {question} is included. Please Only return the boolean value to represent it.
            """,
    )
    chain1 = LLMChain(llm=llm1, prompt=prompt_template1, output_key="urgent_key")
    llm2 = OpenAI(temperature=0.7)
    prompt_template2 = PromptTemplate(
        input_variables=["question", "team_document_docs"],
        template="""
            Based on {team_document_docs}, summarize the {question} into a very short sentence.
            """,
    )
    chain2 = LLMChain(llm=llm2, prompt=prompt_template2, output_key="summarization")
    llm3 = OpenAI(temperature=0.7)
    prompt_template3 = PromptTemplate(
        input_variables=["summarization", "team_document_docs"],
        template="""
            Based on {team_document_docs} and {summarization}, Give me a suggested response.
            """,
    )
    chain3 = LLMChain(llm=llm3, prompt=prompt_template3, output_key="response")

    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3],
        input_variables=["question", "team_document_docs"],
        output_variables=["urgent_key", "summarization", "response"],
        verbose=False,
    )
    result = overall_chain(
        {"question": question, "team_document_docs": team_document_docs}
    )
    return {
        "urgency": "True" == (result["urgent_key"].replace("\n", "").lstrip()),
        "summarization": result["summarization"].replace("\n", "").lstrip(),
        "response": result["response"].replace("\n", "").lstrip(),
    }


"""
Based on {team_document_docs}, give me the classification and urgency of {question} and summarize the {question} into a very short sentence.
Give me a suggested response based on {team_document_docs}
"""


class ConversationChain(LLMChain, BaseModel):
    """Chain to have a conversation and load context from memory.

    Example:
        .. code-block:: python

            from langchain import ConversationChain, OpenAI
            conversation = ConversationChain(llm=OpenAI())
    """

    memory: Any
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""

    input_key: str = "input"  #: :meta private:
    output_key: str = "response"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Use this since so some prompt vars come from history."""
        return [self.input_key]

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        memory_keys = values["memory"].memory_variables
        input_key = values["input_key"]
        if input_key in memory_keys:
            raise ValueError(
                f"The input key {input_key} was also found in the memory keys "
                f"({memory_keys}) - please provide keys that don't overlap."
            )
        prompt_variables = values["prompt"].input_variables
        expected_keys = memory_keys + [input_key]
        if set(expected_keys) != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but got {memory_keys} as inputs from "
                f"memory, and {input_key} as the normal input key."
            )
        return values
