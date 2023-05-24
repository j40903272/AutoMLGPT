import re
import json
import logging
import threading
from typing import Union
from abc import ABC, abstractmethod

from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    # ConversationBufferWindowMemory,
    # ConversationSummaryBufferMemory
)
from langchain.chat_models import ChatOpenAI
from langchain.agents.tools import Tool
from langchain.agents.conversational.output_parser import ConvoOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.conversational_chat.prompt import (
    FORMAT_INSTRUCTIONS as CHAT_FORMAT_INSTRUCTIONS,
)
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser


import functools
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd


from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from sklearn.svm import LinearSVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox
from imblearn.over_sampling import SMOTE


from AutoMLGPT.tools import E2EHyperOptTool, ExploreDataTool, ProcessDataTool
from AutoMLGPT.prompt import AUTOML_CHATGPT_PREFIX, AUTOML_CHATGPT_FORMAT_INSTRUCTIONS, AUTOML_CHATGPT_SUFFIX


logger = logging.getLogger()


class CodingAgent():
    def __init__(self):
        
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="{text}",
                input_variables=["text"],
            )
        )
        chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
        chat = ChatOpenAI(temperature=0)
        self.chain = LLMChain(llm=chat, prompt=chat_prompt_template)#, memory=ConversationBufferMemory())
    @functools.lru_cache(100)
    def run(self, text):
        return self.chain.run(text)
    
    
coding_agent = CodingAgent(chain)



class retryActionTool:
    """Retry when chatgpt response parsing failed."""

    @prompts(
        name="RR",
        description="Do not use it.",
    )
    def inference(self, query):
        return f"Modify the following text in ``` such that it will follow RESPONSE FORMAT INSTRUCTIONS:```{query}```\nDo not use retry tool again."


class RetryConvoOutputParser(ConvoOutputParser):
    ai_prefix: str = "AI"

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            return AgentAction("retry", "", text)
            # raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)


class RobustRetryConvoOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return CHAT_FORMAT_INSTRUCTIONS

    def parse_json(self, text: str):
        try:
            return [
                i
                for i in re.findall(r"\s([{\[].*?[}\]])$", f" {text}", flags=re.DOTALL)
                if i
            ][0]
        except Exception:
            return

    def parse_markdown_code(self, text: str):
        try:
            for i in re.findall(
                r"`{3}([\w]*)\n([\S\s]+?)\n`{3}", text, flags=re.DOTALL
            )[0]:
                if i:
                    tmp = self.parse_json(i)
                    if tmp:
                        return tmp
        except Exception:
            return

    def parse_origin(self, text: str):
        try:
            cleaned_output = text.strip()
            if "```json" in cleaned_output:
                _, cleaned_output = cleaned_output.split("```json")
            if "```" in cleaned_output:
                cleaned_output, _ = cleaned_output.split("```")
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[len("```json"):]
            if cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[len("```"):]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[: -len("```")]
            return cleaned_output.strip()
        except Exception:
            return

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            json_str = self.parse_json(text)
            markdown_str = self.parse_markdown_code(text)
            origin_str = self.parse_origin(text)

            if json_str:
                cleaned_output = json_str
            elif markdown_str:
                cleaned_output = markdown_str
                if cleaned_output.startswith("```json"):
                    cleaned_output = cleaned_output[len("```json"):]
                if cleaned_output.startswith("```"):
                    cleaned_output = cleaned_output[len("```"):]
                if cleaned_output.endswith("```"):
                    cleaned_output = cleaned_output[: -len("```")]
            elif origin_str:
                cleaned_output = origin_str
            else:
                cleaned_output = text.strip()

            response = json.loads(cleaned_output)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except Exception:
            if "action" not in text and "action_input" not in text:
                return AgentFinish({"output": text}, text)
            else:
                logger.warning(f"Not follow format instruction\n{cleaned_output}")
                return AgentAction("RR", "", text)

        

class ConversationBot(ABC):

    def __init__(self):
        seed_everything(0)
        # self.handler = LoggerCallbackHandler()
        # self.async_handler = AsyncLoggerCallbackHandler()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output", return_messages=True
        )
        self.state = []
        self.tools = self.load_tools()

        # retry when llm output parsing error
        retrytool = retryActionTool()
        func = retrytool.inference
        self.tools.append(
            Tool(
                name=func.name,
                description=func.description,
                # coroutine=to_async(func),
                func=func,
            )
        )
        self.output_parser = RobustRetryConvoOutputParser()
        self.agent = self.init_agent()

    def run_text(self, text: str):
        logger.info(f"User: {text}")
        res = self.agent({"input": text})
        response = res["output"]
        self.state += [(text, response)]
        logger.info(
            f"\nProcessed run_text, Input text: {text}\nCurrent state: {self.state}\n"
        )
        return response

    async def arun_text(self, text: str):
        logger.info(f"User: {text}")
        res = await self.agent.acall({"input": text})
        response = res["output"]
        self.state += [(text, response)]
        logger.info(
            f"\nProcessed run_text, Input text: {text}\nCurrent state: {self.state}\n"
        )
        return response

    def _clear(self):
        self.memory.clear

    def init_agent(self):
        input_variables = ["input", "agent_scratchpad", "chat_history", "dataset"]
        return initialize_agent(
            self.tools,
            self.llm,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            max_iterations=5,
            # max_execution_time=60,
            agent_kwargs={
                "prefix": AUTOML_CHATGPT_PREFIX,
                "format_instructions": AUTOML_CHATGPT_FORMAT_INSTRUCTIONS,
                "suffix": AUTOML_CHATGPT_SUFFIX,
                "input_variables": input_variables,
                "output_parser": self.output_parser,
            },
        )

    def load_tools(self):
        tools = []
        X, X_train, X_test, y, y_train, y_test = self.load_artifacts()

        for tool in [
            # pythonShellTool(),
            E2EHyperOptTool(X_train, y_train, X_test, y_test),
            # HyperOptTool(),
            # ConstructSearchTool(),
            # ConstructParamGridTool(),
            # ConstructEstimatorTool(),
            # FinalTool(X, Y),
            ExploreDataTool(status),
            ProcessDataTool(status),
        ]:
            func = tool.inference
            tools.append(
                Tool(
                    name=func.name,
                    description=func.description,
                    # coroutine=to_async(func),
                    func=func,
                )
            )
        logger.info(f"tools: {[i.name for i in tools]}")
        return tools

    def load_artifacts(self):
        data = pd.read_csv("WineQT.csv")
        X = data.drop('quality', axis=1)
        y = data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        return X, X_train, X_test, y, y_train, y_test