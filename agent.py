import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.json import JSONKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.storage.redis import RedisStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.document.chunking.document import DocumentChunking
from agno.reranker.cohere import CohereReranker
from agno.models.cohere import Cohere 
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

#log level debug
# logging.basicConfig(level=logging.DEBUG)



# Initialize knowledge base for products
products_knowledge = JSONKnowledgeBase(
    path="./products-split",
    
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="hrms_products",
        search_type=SearchType.vector,
        embedder=GeminiEmbedder(api_key=os.getenv("GEMINI_API_KEY")),
        reranker=CohereReranker(api_key=os.getenv("COHERE_API_KEY")),
    ),
    num_documents=20,  # Return top 10 results
)


# Initialize Redis storage for session management
redis_storage = RedisStorage(
    prefix="octopus",
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    password=os.getenv("REDIS_PASSWORD")
)


# System instructions for Octopus
OCTOPUS_INSTRUCTIONS = """
You are Octopus, an intelligent HRMS (Human Resource Management System) assistant that helps users find the perfect HR products and solutions for their needs.

Your primary responsibilities:
1. Help users find HRMS products based on their requirements
2. Answer questions about HR, hiring, talent management, and related topics
3. Provide detailed information about products, including features, benefits, and use cases
4. Assist with HR-related problems and provide solutions

When searching for products:
- ALWAYS search your knowledge base first for existing products
- Prioritize products based on their weight/importance (higher weightage = higher priority)
- When multiple products match, present them in order of relevance and weight
- Provide comprehensive details about each product including features, benefits, ideal company sizes, and use cases

If no existing products match the user's criteria:
- Clearly state that no exact matches were found in the product database
- Use your general HR knowledge to provide helpful advice
- If needed, search the web for additional information using the web search tool
- Always indicate when information comes from web search vs. the product database

Product Information Guidelines:
- Present product information in a clear, structured format
- Highlight key features and benefits
- Mention target industries and ideal company sizes
- Include implementation scenarios when relevant
- Note any competitive advantages

Keep your responses very short and very concise.
when listing products, list atleast 3 products.

You can ask the user the right questions like asking for their requirement, company size, industry, etc. Whatever is applicable.

Remember to:
- Be helpful, professional, and knowledgeable
- Provide accurate, detailed responses
- Guide users toward the best solutions for their needs
- Acknowledge when you need to search for additional information

"""


def create_agent(session_id: Optional[str] = None, user_id: str = "default") -> Agent:
    """Create an Octopus agent instance with the given configuration"""
    
    agent = Agent(
        name="Octopus - HRMS Assistant",
        model=Gemini(id=os.getenv("AGENT_MODEL", "gemini-2.5-flash-preview-05-20"), api_key=os.getenv("GEMINI_API_KEY")),
        
        #model=Cohere(id="command-r", api_key=os.getenv("COHERE_API_KEY")),
        instructions=OCTOPUS_INSTRUCTIONS,
        knowledge=products_knowledge,
        tools=[DuckDuckGoTools()],
        storage=redis_storage,
        session_id=session_id,
        user_id=user_id,
        # Enable knowledge search
        search_knowledge=True,
        # Show tool usage
        show_tool_calls=True,
        # Enable markdown formatting
        markdown=True,
        # Add chat history reading capability
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=10,
        # Add datetime to instructions for context
        add_datetime_to_instructions=True,
        # Number of knowledge documents to consider
        debug_mode=True
    )
    
    return agent
import time

import glob


# loaded_files = []
# total_files = len(glob.glob("products-split/*.json"))
# while True:
#     try:
        
#         for file in glob.glob("products-split/*.json"):
        
#             if file not in loaded_files:
#                 products_knowledge.load_document(path=file, recreate=False, skip_existing=True)
#                 loaded_files.append(file)
#             print(file)
#             print(len(loaded_files), "/", total_files)
#         break
#     except Exception as e:
#         print(f"Error loading knowledge base: {e}")
#         print("Attempting to recreate knowledge base...")
#         time.sleep(10)


# exit()
# while True:
#     try:
#         products_knowledge.load(recreate=False, skip_existing=True)
#         print("Knowledge base loaded successfully!")
#         break
#     except Exception as e:
#         print(f"Error loading knowledge base: {e}")
#         print("Attempting to recreate knowledge base...")
#         time.sleep(10)