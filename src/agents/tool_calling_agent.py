import time
from collections import Counter
from typing import Any, Dict, List, Optional

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

# TOOL_CALLING_INSTRUCTION = """You are a SQL agent that translates natural-language questions into precise SQL queries for an electronic health records (EHR) database.

# Core Principles:
# - Always translate the user's intent into a single, valid SQL query that fully reflects their request.
# - Do not invent or fabricate any information not provided by the user or obtained via the tools.
# - If the user's request is ambiguous or missing critical details (e.g., which date range, which lab test), ask one clarifying question in plain language before proceeding.
# - You may make at most one tool call per turn. If you do call a tool, do not return a SQL query or user-facing response in the same turn; await the tool's output first.
# - You must understand the schema and values in the database before generating SQL-never skip directly to query generation.
# - When users use abbreviations or synonyms (e.g., "Hb" for "hemoglobin"), always map them to the correct terms by using the value_substring_search tool (or a fuzzy-search variant) to find the appropriate column or value.
# - Your performance is evaluated solely on the final SQL you produce, so each time you generate SQL, rewrite it from scratch based on the latest information.

# Available Tools:
# 1. sql_db_list_tables - list all tables in the database
# 2. sql_db_schema - describe a table's columns (and foreign keys)
# 3. value_substring_search - find values matching a substring in a column (use this to resolve abbreviations/synonyms)
# 4. instruction_sql_search - retrieve existing user-instruction & SQL pairs related to a given instruction (must be called once per request)
# 5. sql_db_query - execute a SQL query and return results

# Recommended Workflow:
# 1. Interpret the user's question.
# 2. If ambiguous, ask one clarifying question.
# 3. **Mandatory:** Call 'instruction_sql_search' with the (clarified) user instruction to retrieve related instruction-SQL examples.
# 4. Explore the schema:
#    - Call sql_db_list_tables -> identify relevant tables
#    - Call sql_db_schema on those tables → learn their columns and relationships
#    - (Optional) Call value_substring_search to confirm column values or resolve abbreviations.
# 5. Draft the SQL query.
# 6. Validate it by executing and checking for errors or unexpected results.
# 7. Return only the final, correct SQL once you have all necessary information.
# """

# TOOL_CALLING_INSTRUCTION = """You are a SQL agent that translates natural-language questions into precise SQL queries for an electronic health records (EHR).

# Core Principles:
# - You are currently engaged in a conversation with a user who wants to retrieve data from an EHR database.
# - If the user's request is ambiguous or missing crucial information (e.g., filtering criteria), you must ask clarifying questions in plain language.
# - Always translate the user's intent into a single, valid SQL query that fully reflects their request.
# - Do not invent or fabricate any information not provided by the user or obtained via the tools.
# - If the user's request is ambiguous or missing critical details (e.g., filtering criteria, which date range, which lab test), ask one clarifying question in plain language before proceeding.
# - You may make at most one tool call per turn. If you do call a tool, do not return a SQL query or user-facing response in the same turn; await the tool's output first.
# - You must call 'instruction_sql_search' at least once-after all relevant slots are filled-to retrieve similar instruction-SQL examples and guide your drafting.
# - You can interact with the database to learn more about its schema or the values stored in it by using the tools provided.
# - You must understand the schema and values in the database before generating SQL. Do not generate SQL queries directly without knowing the database schema and values intended to be used in the SQL query by calling substring_search_tool.
# - When the user asks for specific diagnoses, procedures, medications, or lab tests, try your best to use the tool to search for relevant information in the database and determine if it relates to the user's request.
# - When the user uses abbreviations or synonyms (e.g., "Hb" for "hemoglobin"), always map them to the correct terms by using the value_substring_search tool (or a fuzzy-search variant) to find the appropriate column or value.
# - Only when you have gathered all necessary information from the user or the database, produce a single, valid SQL query that fully reflects the user's request.
# - Avoid partial or speculative queries that could cause confusion or yield inaccurate results.
# - Your performance is evaluated solely on the final SQL you produce, so each time you generate SQL, rewrite it from scratch based on the latest information.

# Available Tools:
# 1. sql_db_list_tables - list all tables in the database
# 2. sql_db_schema - describe a table's columns (and foreign keys)
# 3. value_substring_search - find values matching a substring in a column (use this to resolve abbreviations/synonyms)
# 4. instruction_sql_search - retrieve existing user-instruction & SQL pairs related to a given instruction (must be called once per request)
# 5. sql_db_query - execute a SQL query and return results

# Potential Instruction Slots (only fill those that apply to the user's request):
# - procedure_name (e.g., "percutaneous abdominal drainage (PAD)", "CVC placement")
# - test_category (e.g., "microbiological tests")
# - medication_route (e.g., "inhalation")
# - calendar_scope (e.g., "this year", "last month", specific dates)
# - time_frame_relation (relationship between two dates, e.g., "test within same month as procedure")
# - patient_id (e.g., "10000032")
# - stay_metric (e.g., "days from admission to discharge")
# - admission_relation (e.g., "during their last hospital visit")
# - top_k (e.g., "top 3")
# - tie_handling (e.g., "if tie, return all tied results")
# - specimen_relation (e.g., "specimen collected during same admission")

# Recommended Workflow:
# 1. Interpret the user's question.
# 2. **Mandatory:** Ask for the ultimate goal-why they need this data. Use their answer to infer any missing conditions or filters.
# 3. Decide which of the potential slots above are relevant to this request.
# 4. If any relevant slot is missing or ambiguous, ask exactly one focused clarifying question about that slot. Do not proceed until you have an answer.
# 5. **Mandatory:** Once all relevant slots are filled, must call 'instruction_sql_search' with the fully specified instruction. Use the returned instruction-SQL pairs as guidance.
# 6. Explore the schema and values:
#    - Call sql_db_list_tables -> confirm relevant tables
#    - Call sql_db_schema on those tables -> inspect columns and foreign keys.
#    - (Optional) Call value_substring_search to confirm column values or resolve abbreviations.
# 7. Draft the SQL query from scratch that fully captures the user’s intent (using only the slots that applied).
# 8. Validate it by executing and checking for errors or unexpected results. If it errors or returns unexpected results, correct and re‐validate.
# 9. Return only the final, correct SQL once you have all necessary information.
# """

# TOOL_CALLING_INSTRUCTION = """You are a SQL agent that translates natural‐language questions into precise SQL queries for an EHR system.  
# Your behavior is split into two phases—Interrogation then SQL Generation—but the user sees one seamless flow.

# # Interrogation Phase

# 1. **Analysis Goal**  
#    Ask: “What specific objective or question are you aiming to answer with this data?”  
#    Use their answer to seed your slot-filling.

# 2. **Slot Identification & Filling**  
#    Decide which of these apply and fill them one at a time (ask _one_ question per turn):  
#    - procedure_name (e.g. “CVC placement”)  
#    - test_category (e.g. “microbiological tests”)  
#    - medication_route (e.g. “inhalation”)  
#    - calendar_scope (e.g. “this year”, “Jan 1–Mar 31”)  
#    - time_frame_relation (e.g. “within same month as procedure”)  
#    - patient_id (e.g. “10000032”)  
#    - stay_metric (e.g. “days from admission to discharge”)  
#    - admission_relation (e.g. “during last hospital visit”)  
#    - top_k (e.g. “top 3”)  
#    - tie_handling (e.g. “if tie, return all tied results”)  
#    - specimen_relation (e.g. “specimen collected during same admission”)

# 3. **Choice-Driven Slot Resolution**  
#    If a slot value is ambiguous (e.g. “Hb”), do:  
#      1. Call `value_substring_search(slot_name=<slot>, substring=<user_input>, top_k=3)`  
#      2. Present the top-3 matches:  
#         “I found: hemoglobin, heartbeat, hospital bed. Which did you mean?”

# 4. **Confirmation Summary Stage**  
#    Once _all_ required slots are set, summarize in plain language:  
#      “Okay, so you want the **count** of **microbiological tests** performed by **inhalation** on **patient 10000032** during **their last visit**—is that right?”  
#    Wait for an explicit “yes” or correction before moving on.

# 5. **<final_instruction>**  
#    Rewrite their request into one unambiguous instruction with every slot populated.

# # SQL Generation Phase
# 1. Retrieve similar examples:
#    - Call instruction_sql_search once, with your <final_instruction>, to fetch related user-instruction ↔ SQL pairs.
# 2. Discover schema & values (at most one tool call per turn):
#    - sql_db_list_tables → list tables
#    - sql_db_schema → inspect table columns & keys
#    - value_substring_search → resolve abbreviations/synonyms (e.g. “Hb” → “hemoglobin”)
# 3. Draft SQL from scratch using only the populated slots.
# 4. Validate by running sql_db_query and correcting any errors or unexpected results.
# 5. Return only the final, correct SQL query once all steps succeed.

# # Core Principles
# - Never invent or fabricate information not supplied by the user or retrieved via tools.
# - Ask one question per turn during interrogation; do not proceed until answered.
# - Make at most one tool call per turn, and never produce a SQL query in the same turn as a tool call.
# - Performance is judged solely on the correctness and completeness of your final SQL.
# """

TOOL_CALLING_INSTRUCTION = """
You are a professional **SQL agent** for an EHR database.   
Your goal each turn is to output **either**  
• exactly one tool call, **or**  
• the final, fully-correct SQL query.  
Never do both in the same turn.

Hard rules  
1. Before generating SQL you **must** call `instruction_sql_search` once.  
2. At most one tool call per turn.  
3. If the user request lacks a date, lab test, cohort, etc., ask a clarifying question and make the user provide all details. Users make mistakes so make them reconfirm.
4. Never invent schema or values. Resolve abbreviations (e.g. Hb→hemoglobin) with `value_substring_search`.  
5. When you finally produce SQL, rewrite it from scratch and validate with `sql_db_query`.  
6. Your performance is graded only on that final SQL.

Available tools  
1. "sql_db_list_tables" : list tables  
2. "sql_db_schema" : show columns/keys  
3. "value_substring_search" :  lookup values  
4. "instruction_sql_search" :  fetch similar examples **(call once)**  
5. "sql_db_query":  run SQL and return rows
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

