from langchain_community.document_loaders import TextLoader
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from typing_extensions import List, TypedDict, Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

class Search(TypedDict):
    """Search query."""
    query: Annotated[str, ..., "Search query to run."]

class RAGChatbot:
    def __init__(self, folder_path="/Users/doblado/Datathon2025/altas/"):
        load_dotenv()
        self.api_key = os.getenv("DEDALUS_API_KEY")
        self.folder_path = folder_path
        self.memory = MemorySaver()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://litellm.dccp.pbu.dedalus.com",
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            temperature=0.2
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Set up documents and vectorstore
        self.setup_vectorstore()
        
        # Set up the graph
        self.graph = self.build_graph()
    
    def load_documents(self, folder_path):
        """Load all text documents from a directory"""
        documents = []
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading documents: {e}")
        return documents
    
    def setup_vectorstore(self):
        """Set up the vector store with documents"""
        documents = self.load_documents(self.folder_path)
        if not documents:
            # Create a dummy document if no documents are found
            documents = [Document(page_content="No medical summaries found")]
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1400,
            chunk_overlap=300,
            length_function=len
        )
        self.splits = text_splitter.split_documents(documents)
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(self.splits, self.embeddings)
    
    def retrieve(self, query):
        """Retrieve information related to a query"""
        try:
            retriever = self.vectorstore.as_retriever()
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            compressed_docs = compression_retriever.invoke(query)
            return compressed_docs
        except Exception as e:
            print(f"Error in retrieve: {e}")
            return []
    
    def query_or_respond(self, state):
        """Generate tool call for retrieval or respond"""
        # Create a proper langchain Tool
        retrieve_tool = Tool(
            name="retrieve",
            func=self.retrieve,
            description="Retrieve information related to a query."
        )
        
        llm_with_tools = self.llm.bind_tools([retrieve_tool])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    def generate(self, state):
        """Generate answer"""
        # Get context from tool messages
        docs_content = ""
        try:
            # Find the tool messages
            tool_messages = [m for m in state["messages"] if m.type == "tool"]
            
            if tool_messages:
                # Different tools might return different formats, so we need to handle them flexibly
                tool_result = tool_messages[-1]
                
                # Extract document content based on the result structure
                if hasattr(tool_result, 'content') and tool_result.content:
                    docs_content = tool_result.content
                elif hasattr(tool_result, 'artifact') and tool_result.artifact:
                    # Handle different types of artifacts
                    if isinstance(tool_result.artifact, list):
                        # If it's a list of documents
                        docs_texts = []
                        for doc in tool_result.artifact:
                            if hasattr(doc, 'page_content'):
                                docs_texts.append(doc.page_content)
                        docs_content = "\n\n".join(docs_texts)
                    elif hasattr(tool_result.artifact, 'page_content'):
                        docs_content = tool_result.artifact.page_content
                    else:
                        docs_content = str(tool_result.artifact)
        except Exception as e:
            print(f"Error processing tool messages: {e}")
            docs_content = "Error retrieving context."
        
        if not docs_content:
            docs_content = "No relevant information found."
            
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. Respond in Spanish."
            "\n\n"
            f"{docs_content}"
        )
        
        # Get conversation messages
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages
        
        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}
    
    def build_graph(self):
        """Build the graph for the RAG system"""
        # Create a proper langchain Tool
        retrieve_tool = Tool(
            name="retrieve",
            func=self.retrieve,
            description="Retrieve information related to a query."
        )
        
        tools = ToolNode([retrieve_tool])
        
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate", self.generate)
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        return graph_builder.compile(checkpointer=self.memory)
    
    def refresh_vectorstore(self):
        """Refresh the vector store with updated documents"""
        self.setup_vectorstore()
    
    def process_query(self, query, thread_id="1"):
        """Process a single query through the RAG system"""
        config = {"configurable": {"thread_id": thread_id}}
        result = []
        
        try:
            for event in self.graph.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
                config=config,
            ):
                if "messages" in event and event["messages"]:
                    result.append(event["messages"][-1].content)
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
        
        return result[-1] if result else "No response generated"