from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Optional
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_experimental.agents import create_csv_agent

class AgentState(TypedDict):
    task: str
    info: List[str]
    plan: str
    draft: str
    critique: str
    safety: Optional[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]

# Prompt templates
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline for a medical health record. \
Write such an outline with the provided data information, if some information is missed don't include it. Give an outline of the medical record along with any relevant notes \
or instructions for the sections. Don't include the information that could not be relevant. Suggest to divide the draft in header + body + footer."""

WRITER_PROMPT = """You are an medical assistant tasked with writing excellent health record resumes.\
Generate the best resume possible for the user's request and the initial outline. \
Generate it in Spanish. \
If the user provides critique, respond with a revised version of your previous attempts. \
Based on all the resume, add a final wrapper highlighting the most important points and giving recommendations for the ongoing treatment process. \
Use a formal medical style. \
Utilize all the information below as needed, if some information is missed avoid mentioning that point: 

------

{info}"""

SAFETY_PROMPT = """You are a medical safety checker. Identify any potentially harmful or incorrect medical advice. \
    Also check for legal compliance issues in the generated health record resume"""

REFLECTION_PROMPT = """You are a doctor with years of experience evaluating a health record resume. \
Generate critique and recommendations for the provided resume. \
Provide detailed recommendations, including requests for length, depth, style, factually and check for possible compliance issues."""

class MedicalAgent:
    def __init__(self, api_key=None, base_url=None, model=None, csv_path=None):
        # Load environment variables if API key not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("DEDALUS_API_KEY")
            
        # Set default values if not provided
        self.base_url = base_url or "https://litellm.dccp.pbu.dedalus.com"
        self.model = model or "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.csv_path = csv_path
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0.2
        )
        
        print("Creando un analista de datos")
        try:
            self.csv_agent = create_csv_agent(
                self.llm,
                self.csv_path,
                verbose=False,  # Set to False to reduce overhead
                allow_dangerous_code=True
            )
            print("-----")
            print("Analista creado correctamente, ya casi estamos!")
        except Exception as e:
            print(f"Error creando el CSV agent: {str(e)}")
            self.csv_agent = None
        
        # Build the graph
        self.graph = self._build_graph()
    
    def extractor_node(self, state: AgentState):
         try:
            if self.csv_agent is None:
                return {"info": "Error: CSV agent not initialized properly"}
                
            query = f"Search inside the dataframe and provide all the information available related to the given query. Mention if you detect possible anomalies in the data: {state['task']}"
            print("-----")
            print("Procesando y analizando tu petición...")
            
            # Use the pre-created CSV agent
            response = self.csv_agent.invoke(query)
            return {"info": response}
         except Exception as e:
            return {"info": f"Error extracting information: {str(e)}"}

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT), 
            HumanMessage(f"Query: {state['task']}\n\nExtracted Medical Data:\n{state['info']}")
        ]
        response = self.llm.invoke(messages)
        return {"plan": response.content}

    def generation_node(self, state: AgentState):
        print("----")
        print("Realizando iteraciones sobre el resumen para obtener el mejor resultado...")
        info = "\n\n".join(state['info'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [
            SystemMessage(
                content=WRITER_PROMPT.format(info=info)
            ),
            user_message
            ]
        response = self.llm.invoke(messages)
        return {
            "draft": response.content, 
            "revision_number": state.get("revision_number", 1) + 1,
        }
        
    def safety_check_node(self, state: AgentState):
        messages = [
            SystemMessage(content=SAFETY_PROMPT),
            HumanMessage(content=state['draft'])
        ]
        response = self.llm.invoke(messages)
        return {"safety_issues": response.content}
        
    def reflection_node(self, state: AgentState):
        previous_critique = state.get("critique", "")
        messages = [
            SystemMessage(content=REFLECTION_PROMPT), 
            HumanMessage(content=f"Previous critique: {previous_critique}\n\nCurrent draft: {state['draft']}")
        ]
        response = self.llm.invoke(messages)
        return {"critique": response.content}

    def should_continue(self, state: AgentState):
        if state["revision_number"] >= state["max_revisions"]:
            return END 
        return "security"
    
    def _build_graph(self):
        memory = MemorySaver()
        builder = StateGraph(AgentState)

        builder.add_node("extractor", self.extractor_node)
        builder.add_node("planner", self.plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("security", self.safety_check_node)
        builder.add_node("reflect", self.reflection_node)

        # Set entry point
        builder.set_entry_point("extractor")

        # Add edges
        builder.add_edge("extractor", "planner")
        builder.add_conditional_edges(
            "generate", 
            self.should_continue, 
            {END: END, "security": "security"}
        )
        builder.add_edge("planner", "generate")
        builder.add_edge("security", "reflect")   
        builder.add_edge("reflect", "generate")   

        return builder.compile(checkpointer=memory)
    
    def process_patient(self, patient_query, max_revisions, thread):
        
        
        results = []
        
        for s in self.graph.stream({
            'task': patient_query,
            "revision_number": 1,
            "max_revisions": max_revisions,
        }, thread):
            results.append(s)
            
        return results[-1] if results else None
