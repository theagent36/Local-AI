import os
import ollama
from document_processor import DocumentProcessor

# 1. Define your tools as standard Python functions.
# The docstrings and type hints are critical—Ollama parses them automatically 
# to tell the LLM exactly how and when to use this tool.
def get_weather(city: str) -> str:
    """Get the current weather conditions for a specific city.
    
    Args:
        city: The name of the city (e.g., 'Paris', 'Tokyo')
    """
    # Mock data for demonstration
    mock_database = {
        'Paris': '22°C and sunny',
        'Tokyo': '18°C and raining',
        'New York': '15°C and cloudy'
    }
    return mock_database.get(city, f"Weather data not found for {city}")

def read_notes_or_document(file_path: str, max_pages: int = 5) -> str:
    """Read and transcribe handwritten notes or documents from an image or PDF file.
    
    Args:
        file_path: The local path to the image (e.g., .png, .jpg, .jpeg) or PDF file.
        max_pages: Maximum number of pages to read (default 5 to preserve speed and context).
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at '{file_path}'."
    
    try:
        processor = DocumentProcessor(vision_model="moondream")
        docs = processor.process_file(file_path, file_name=os.path.basename(file_path))
        
        extracted = []
        for d in docs[:max_pages]:
            page = d.metadata.get("page", 1)
            doc_type = d.metadata.get("type", "document")
            extracted.append(f"--- Page {page} ({doc_type}) ---\n{d.page_content}")
            
        if len(docs) > max_pages:
            extracted.append(f"... [Note: Showing first {max_pages} of {len(docs)} total pages]")

        return "\n\n".join(extracted)
    except Exception as e:
        return f"Error transcribing '{file_path}': {str(e)}"

# Map function names to actual Python function objects
available_functions = {
    'get_weather': get_weather,
    'read_notes_or_document': read_notes_or_document,
}

tools_list = [get_weather, read_notes_or_document]

def run_agent(prompt: str):
    # Initialize the conversation history
    messages = [{'role': 'user', 'content': prompt}]

    print(f"User: {prompt}\n")
    print("Agent is thinking...")

    # 2. Call the model and provide the list of tools
    response = ollama.chat(
        model='llama3.2',
        messages=messages,
        tools=tools_list
    )

    # Append the model's response to the history
    messages.append(response.message)

    # 3. Check if the model decided to call a tool
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            func_name = tool_call.function.name
            func_args = tool_call.function.arguments
            
            print(f"[Executing Tool: {func_name} with args: {func_args}]")

            # Execute the actual Python function
            if func_name in available_functions:
                function_to_call = available_functions[func_name]
                tool_result = function_to_call(**func_args)
                
                print(f"[Tool Result Preview: {str(tool_result)[:150]}...]\n")

                # 4. Feed the tool's output back to the model
                messages.append({
                    'role': 'tool',
                    'name': func_name,
                    'content': str(tool_result)
                })
            else:
                print(f"Error: Model requested unknown tool {func_name}")

        # 5. Call the model one last time so it can read the tool results 
        # and generate a natural language response for the user.
        final_response = ollama.chat(
            model='llama3.2',
            messages=messages,
            tools=tools_list
        )
        
        print("Final Output:")
        print(final_response.message.content)

    else:
        # If no tools were called, just print the direct response
        print(response.message.content)

if __name__ == "__main__":
    # Test agent with weather tool
    run_agent("What is the weather like in Tokyo?")