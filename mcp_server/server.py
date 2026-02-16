
from mcp.server.fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("MyCustomLLM-Server")

@mcp.tool()
async def generate_text(prompt: str, max_tokens: int = 100) -> str:
    """
    Calls the custom local LLM to generate a response for a given prompt.
    """
    # Inference logic using your trained model
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    output = model.generate(input_ids, max_new_tokens=max_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Runs the server over stdio for local AI assistant integration
    mcp.run(transport="stdio")
