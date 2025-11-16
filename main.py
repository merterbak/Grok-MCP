import os
from dotenv import load_dotenv
from src import main, mcp

load_dotenv()

if __name__ == "__main__":
        
    if not os.getenv("XAI_API_KEY"):
        print(" Warning: XAI_API_KEY not found in environment.")
        print("Please set your API key in .env file or export it: export XAI_API_KEY='your_api_key' ")
        print("Starting server will fail on first API call")
    else:
        print("XAI_API_KEY found")
        print("Started MCP-Grok server")
    
    main()

