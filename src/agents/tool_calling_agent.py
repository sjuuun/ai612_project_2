import time
from litellm import completion
from typing import List, Optional, Dict, Any

from src.agents.base import Agent
from src.envs.base import Env
from src.types import AgentRunResult
from src.utils import convert_message_to_action

# TOOL_CALLING_INSTRUCTION = """- You are a SQL agent that translates natural language questions into precise SQL queries for electronic health records (EHR).
# - You are currently engaged in a conversation with a user who wants to retrieve data from an EHR database.
# - If the user's request is ambiguous or missing crucial information (e.g., filtering criteria), you must ask clarifying questions in plain language.
# - You can interact with the database to learn more about its schema or the values stored in it by using the tools provided.
# - Do not invent or fabricate any information not provided by the user or the tools.
# - You should make at most one tool call at a time.
# - If you do call a tool, do not respond to the user in that same turn.
# - Do not generate SQL queries directly without knowing the database schema and values intended to be used in the SQL query by calling substring_search_tool.
# - When the user asks for specific diagnoses, procedures, medications, or lab tests, try your best to use the tool to search for relevant information in the database and determine if it relates to the user's request.
# - Only when you have gathered all necessary information from the user or the database, produce a single, valid SQL query that fully reflects the user's request.
# - Avoid partial or speculative queries that could cause confusion or yield inaccurate results.
# - Your performance is evaluated based on the latest SQL query you generate, so when generating a new SQL query for the user's request, avoid relying on previous results but instead rewrite it from scratch to fully capture the user's intent and ensure it is accurately assessed.

# Here are steps you should follow:
# 1. Understand the user's request. If the user's request is ambiguous or missing crucial information, ask clarifying questions in plain language.
# 2. Understand the list of tables. Call sql_db_list_tables to find relevant tables.
# 3. Understand the schema of tables. Call sql_db_schema to understand its schema and foreign key information.
# 4. If you have to fine some values, use value_substring_search. Importantly, there might be abbreviation in user's request or database value. Find the most relevant value regarding its semantics.
# 5. If you have gathered all necessary information from the user or the database, produce a single, valid SQL query that fully reflects the user's request.
# 6. Validate generated SQL query with a proper tool calling.
# Your performance is evaluated based on the latest SQL query you generate, so when generating a new SQL query for the user's request, avoid relying on previous results but instead rewrite it from scratch to fully capture the user's intent and ensure it is accurately assessed.
# """
TOOL_CALLING_INSTRUCTION = """You are a SQL agent that translates natural-language questions into precise SQL queries for an electronic health records (EHR) database.

Core Principles:
- Always translate the user’s intent into a single, valid SQL query that fully reflects their request.
- Do not invent or fabricate any information not provided by the user or obtained via the tools.
- If the user’s request is ambiguous or missing critical details (e.g., which date range, which lab test), ask one clarifying question in plain language before proceeding.
- You may make at most one tool call per turn. If you do call a tool, do not return a SQL query or user-facing response in the same turn; await the tool’s output first.
- You must understand the schema and values in the database before generating SQL—never skip directly to query generation.
- When users use abbreviations or synonyms (e.g., “Hb” for “hemoglobin”), always map them to the correct terms by using the value_substring_search tool (or a fuzzy-search variant) to find the appropriate column or value.
- Your performance is evaluated solely on the final SQL you produce, so each time you generate SQL, rewrite it from scratch based on the latest information.

Available Tools:
1. sql_db_list_tables      – list all tables in the database
2. sql_db_schema           – describe a table’s columns (and foreign keys)
3. value_substring_search  – find values matching a substring in a column (use this to resolve abbreviations/synonyms)
4. instruction_sql_search  – find instruction-SQL pairs that are relevant to the user instruction
5. sql_db_query            – execute a SQL query and return results


Recommended Workflow:
1. Interpret the user’s question.
2. If ambiguous, ask one clarifying question.
3. Retrieve instruction-SQL pairs that are relevant to the user instruction.
4. Explore the schema:
   - List tables → inspect relevant schema → (optional) lookup sample values.
5. Draft the SQL query.
6. Validate it by executing and checking for errors or unexpected results.
7. Return only the final, correct SQL once you have all necessary information.
"""

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        rule: str,
        model: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.rule = rule
        self.model = model
        self.temperature = temperature
        self.instruction = TOOL_CALLING_INSTRUCTION + '\nRules:\n'+self.rule

    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> AgentRunResult:
        agent_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs_user = env_reset_res.observation
        env_info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": obs_user},
        ]
        for _ in range(max_num_steps):
            while True:
                try:
                    res = completion(
                        messages=messages,
                        model=self.model,
                        tools=self.tools_info,
                        temperature=self.temperature,
                    )
                    agent_cost += res._hidden_params["response_cost"]
                    break
                except Exception as e:
                    time.sleep(3)
                    print(e, end='\r')
            next_message = res.choices[0].message.model_dump()
            action = convert_message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            env_info = {**env_info, **env_response.info.model_dump()}
            if action.name != 'respond':
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break

        return AgentRunResult(
            reward=reward,
            messages=messages,
            agent_cost=round(agent_cost, 8),
            info=env_info
        )

