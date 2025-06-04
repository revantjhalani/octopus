import asyncio
import json
import os
from typing import Optional, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import create_agent

# Load environment variables
load_dotenv()


# Initialize FastAPI app
app = FastAPI(title="Octopus HRMS Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str




@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the chat interface"""
    with open('index.html', 'r') as f:
        return f.read()
    



@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    
    async def generate():
        try:
            # Create or get agent
            agent = create_agent(
                session_id=request.session_id,
                user_id=request.user_id
            )
            
            # Send session ID first
            session_data = {
                "session_id": agent.session_id or str(uuid4()),
                "type": "session"
            }
            yield f"data: {json.dumps(session_data)}\n\n"
            
            # Stream the response using agent.arun with stream=True
            run_response = await agent.arun(
                message=request.message,
                stream=True,
                stream_intermediate_steps=False
            )
            
            # Stream each chunk of the response
            async for chunk in run_response:
                if hasattr(chunk, 'content') and chunk.content:
                    data = {
                        "content": chunk.content,
                        "type": "content"
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "type": "error"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )






if __name__ == "__main__":
    
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
    

    
    from agno.playground import Playground, serve_playground_app
    agent = create_agent()
    app = Playground(agents=[agent]).get_app()
    serve_playground_app(app)
    
