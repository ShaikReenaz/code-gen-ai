import streamlit as st
import random
import time
from datetime import datetime
import requests
import json
import pytesseract
from PIL import Image
import io
import os

# Set Tesseract path explicitly
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error(f"Tesseract not found at {tesseract_path}")

# Set Tesseract path (adjust this for your system)
# Common Windows paths:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
# Uncomment and modify the line above if Tesseract is not in PATH

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR - optimized for performance with caching."""
    try:
        # Check if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            return "Error: Tesseract OCR is not installed or not in your PATH. Please install Tesseract OCR first."
        
        # Create a cache key based on image hash
        import hashlib
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
        
        # Check cache first
        if img_hash in st.session_state.ocr_cache:
            return st.session_state.ocr_cache[img_hash]
        
        # Optimize image for faster OCR processing
        # Resize large images to reduce processing time
        max_size = 2000  # Maximum dimension
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale for better OCR results
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply minimal preprocessing for speed
        # Use faster OCR configuration
        custom_config = r'--oem 3 --psm 6 -c tessedit_do_ocr=1'
        
        # Use Tesseract to extract text with optimized settings
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Check if any text was extracted
        if not text.strip():
            result = "No text could be extracted from the image. Please ensure the image contains clear, readable text."
        else:
            result = text.strip()
        
        # Cache the result
        st.session_state.ocr_cache[img_hash] = result
        
        return result
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3",
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=data, stream=True)
        full_reply = ""

        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode())
                    part = json_line.get("response", "")
                    full_reply += part
                except json.JSONDecodeError:
                    pass

        return full_reply.strip()

    except Exception as e:
        return f"⚠️ Error contacting Ollama: {str(e)}"


def process_image_prompt(prompt, image):
    """Process image-related prompts and return appropriate response."""
    if image is None:
        return "Please upload an image first before asking me to process it."
    
    prompt_lower = prompt.lower()
    
    # Check for text extraction requests
    if any(keyword in prompt_lower for keyword in ["extract text", "get text", "ocr", "read text", "what text", "text from image"]):
        with st.spinner('Extracting text from image...'):
            extracted_text = extract_text_from_image(image)
            if extracted_text and not extracted_text.startswith("Error") and not extracted_text.startswith("No text"):
                st.session_state.extracted_text = extracted_text
                return f"I've extracted the following text from your image:\n\n```\n{extracted_text}\n```\n\nWhat would you like me to do with this text?"
            else:
                return f"Unable to extract text from the image: {extracted_text}"
    
    # Check for code analysis requests
    elif any(keyword in prompt_lower for keyword in ["correct", "fix", "debug", "analyze code", "review code", "improve code"]):
        # First extract text if not already done
        if not st.session_state.get('extracted_text'):
            with st.spinner('Extracting text from image first...'):
                extracted_text = extract_text_from_image(image)
                if extracted_text and not extracted_text.startswith("Error") and not extracted_text.startswith("No text"):
                    st.session_state.extracted_text = extracted_text
                else:
                    return f"Unable to extract text from the image for code analysis: {extracted_text}"
        
        if st.session_state.get('extracted_text'):
            code_text = st.session_state.extracted_text
            analysis_prompt = f"Please analyze and correct this code if needed:\n\n```\n{code_text}\n```\n\nUser request: {prompt}"
            return ask_ollama(analysis_prompt)
    
    # Check for output execution requests
    elif any(keyword in prompt_lower for keyword in ["run", "execute", "output", "show output", "what is the output"]):
        # First extract text if not already done
        if not st.session_state.get('extracted_text'):
            with st.spinner('Extracting text from image first...'):
                extracted_text = extract_text_from_image(image)
                if extracted_text and not extracted_text.startswith("Error") and not extracted_text.startswith("No text"):
                    st.session_state.extracted_text = extracted_text
                else:
                    return f"Unable to extract text from the image for execution: {extracted_text}"
        
        if st.session_state.get('extracted_text'):
            code_text = st.session_state.extracted_text
            execution_prompt = f"Please analyze this code and explain what the output would be:\n\n```\n{code_text}\n```\n\nUser request: {prompt}"
            return ask_ollama(execution_prompt)
    
    # Check for example/code writing requests
    elif any(keyword in prompt_lower for keyword in ["example", "similar", "write", "code", "topic"]):
        # First extract text if not already done
        if not st.session_state.get('extracted_text'):
            with st.spinner('Extracting text from image first...'):
                extracted_text = extract_text_from_image(image)
                if extracted_text and not extracted_text.startswith("Error") and not extracted_text.startswith("No text"):
                    st.session_state.extracted_text = extracted_text
                else:
                    return f"Unable to extract text from the image for analysis: {extracted_text}"
        
        if st.session_state.get('extracted_text'):
            code_text = st.session_state.extracted_text
            example_prompt = f"Based on this code/topic from the image:\n\n```\n{code_text}\n```\n\n{prompt}"
            return ask_ollama(example_prompt)
    
    # General image analysis
    elif any(keyword in prompt_lower for keyword in ["analyze", "describe", "explain", "what do you see", "what is in"]):
        analysis_prompt = f"Please analyze this image and respond to: {prompt}. The image contains text/code that has been extracted."
        if st.session_state.get('extracted_text'):
            analysis_prompt += f"\n\nExtracted text/code:\n{st.session_state.extracted_text}"
        return ask_ollama(analysis_prompt)
    
    # If no specific image processing request, treat as general prompt with image context
    else:
        context_prompt = f"I have an uploaded image. {prompt}"
        if st.session_state.get('extracted_text'):
            context_prompt += f"\n\nThe image contains this text/code:\n{st.session_state.extracted_text}"
        return ask_ollama(context_prompt)


def ask_ollama_stream(prompt, message_placeholder=None):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3",
        "prompt": prompt
    }

    try:
        # connection timeout 5s, read timeout 120s
        response = requests.post(url, json=data, stream=True, timeout=(5, 120))
        full_reply = ""

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_line = json.loads(line)
                    part = json_line.get("response", "")
                except Exception:
                    # If not JSON, treat as raw text
                    part = line

                if part:
                    full_reply += part
                    if message_placeholder is not None:
                        # Update UI incrementally as we receive chunks
                        message_placeholder.markdown(full_reply + "▌")

        return full_reply.strip()

    except requests.exceptions.ConnectTimeout:
        return "⚠️ Connection timed out while contacting Ollama."
    except requests.exceptions.ReadTimeout:
        return "⚠️ Read timed out while waiting for Ollama response."
    except Exception as e:
        return f"⚠️ Error contacting Ollama: {str(e)}"


# Page configuration
st.set_page_config(
    page_title="CODE GENAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        padding: 0;
    }
    
    .stChatMessage {
        max-width: 80%;
        margin: 0.5rem auto;
        padding: 1rem;
        border-radius: 1rem;
    }
    
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        background-color: #343541;
        color: white;
        border: 1px solid #565869;
    }
    
    [data-testid="stSidebar"] .stSelectbox div div {
        background-color: #343541;
        color: white;
    }
    
    .chat-container {
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding: 1rem;
    }
    
    .user-message {
        margin-left: auto;
        background-color: #343541;
        color: white;
    }
    
    .assistant-message {
        margin-right: auto;
        background-color: #444654;
        color: white;
    }
    
    /* Style for file attachment buttons */
    button[kind="secondary"] {
        text-align: left !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid #565869 !important;
        background-color: #343541 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        cursor: pointer !important;
    }
    
    button[kind="secondary"]:hover {
        background-color: #40414f !important;
        border-color: #71747d !important;
    }
</style>
""", unsafe_allow_html=True)

def save_current_chat():
    """Save the current chat to the chat history. If no current_chat_id exists, create one."""
    # Only save if there are messages to persist
    if 'messages' in st.session_state and st.session_state.messages:
        # Ensure we have a chat id to index the history
        if 'current_chat_id' not in st.session_state or not st.session_state.current_chat_id:
            st.session_state.chat_counter += 1
            st.session_state.current_chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.chat_history[st.session_state.current_chat_id] = {
            'messages': st.session_state.messages.copy(),
            'created': st.session_state.chat_started
        }
        return True
    return False

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_started" not in st.session_state:
    st.session_state.chat_started = datetime.now()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
if "ocr_cache" not in st.session_state:
    st.session_state.ocr_cache = {}
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Enhanced sidebar
with st.sidebar:
    st.title("🤖 CODE GENAI")
    st.markdown("---")
    
    # Model configuration
    st.subheader("Model Configuration")
    model = st.selectbox(
        "Model:",
        ["GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo", "Claude-3", "Custom"],
        index=0
    )
    
    st.markdown("---")
    
    # Chat management
    st.subheader("Chat Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_started = datetime.now()
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat"):
            # Simple export functionality
            chat_text = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in st.session_state.messages
            ])
            st.download_button(
                "Download Chat",
                chat_text,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
    
    st.markdown("---")
    
    # New Chat
    st.subheader("New Chat")
    if st.button("➕ New Chat", use_container_width=True):
        if 'current_chat_id' in st.session_state and st.session_state.current_chat_id and 'messages' in st.session_state and st.session_state.messages:
            st.session_state.chat_history[st.session_state.current_chat_id] = {
                'messages': st.session_state.messages.copy(),
                'created': st.session_state.chat_started
            }
        
        # Create new chat
        st.session_state.chat_counter += 1
        st.session_state.current_chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.messages = []
        st.session_state.chat_started = datetime.now()
        st.rerun()
    
    st.markdown("---")
    
    # Chat History
    st.subheader("Chat History")
    
    if st.session_state.chat_history:
        for chat_id, chat_data in st.session_state.chat_history.items():
            messages = chat_data.get('messages', [])
            if messages:
                # Get first user message as title
                title = "New Chat"
                for msg in messages:
                    if msg['role'] == 'user':
                        title = msg['content'][:30] + "..." if len(msg['content']) > 30 else msg['content']
                        break
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"💬 {title}", key=f"load_{chat_id}", use_container_width=True):
                        # Save current chat if it exists and has messages
                        save_current_chat()
                        
                        # Load selected chat
                        if chat_id in st.session_state.chat_history:
                            chat_to_load = st.session_state.chat_history[chat_id]
                            st.session_state.messages = chat_to_load.get('messages', []).copy()
                            st.session_state.chat_started = chat_to_load.get('created', datetime.now())
                        else:
                            st.session_state.messages = []
                            st.session_state.chat_started = datetime.now()
                        
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                        del st.session_state.chat_history[chat_id]
                        if st.session_state.current_chat_id == chat_id:
                            st.session_state.messages = []
                            st.session_state.current_chat_id = None
                        st.rerun()
    else:
        st.caption("No chat history. Start a new chat!")
    
    st.markdown("---")
# Main chat interface
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("💬 CODE GENAI")
    st.caption("Experience the power of AI conversation")
    
        # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Store uploaded image in session state
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.session_state.last_uploaded_file = uploaded_file.name
        
        # Display uploaded image info
        st.info(f"Image uploaded: {uploaded_file.name} ({image.size[0]}x{image.size[1]} pixels)")
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.success("Image uploaded successfully! You can now ask me to extract text, analyze code, or perform other tasks with this image.")

    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Add timestamp for messages
                st.caption(f"Message {i+1}")

# Chat input at the bottom of the page
with st.container():
    st.markdown("---")

    # Enhanced chat input with options: input + regenerate
    col1, col2 = st.columns([4, 1])

    with col1:
        prompt = col1.chat_input(
            "Message ChatGPT...",
            key="chat_input"
        )

    with col2:
        if st.button("🔄 Regenerate"):
            # Find last user message
            last_user_idx = None
            for i in range(len(st.session_state.messages)-1, -1, -1):
                if st.session_state.messages[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx is None:
                # No user message to regenerate from
                st.warning("No user message found to regenerate.")
            else:
                # Truncate any assistant replies after the last user message
                st.session_state.messages = st.session_state.messages[: last_user_idx + 1]
                save_current_chat()

                # Regenerate assistant response for the last user prompt
                user_prompt = st.session_state.messages[last_user_idx]["content"]

                # Stream the assistant response and append it
                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown("*Regenerating response...*")
                        response = ask_ollama_stream(user_prompt, message_placeholder)
                        message_placeholder.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                save_current_chat()
                st.rerun()
# Handle user input
if prompt:
    # Ensure we have a current chat ID
    if not st.session_state.current_chat_id:
        st.session_state.chat_counter += 1
        st.session_state.current_chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.chat_started = datetime.now()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Immediately save the updated chat
    save_current_chat()
    
    # Display user message immediately
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Generate and display assistant response
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate AI thinking with more realistic responses
            thinking_phrases = [
                "Let me think about that...",
                "Processing your question...",
                "Analyzing your request...",
                "Generating response..."
            ]
            
            message_placeholder.markdown(f"*{random.choice(thinking_phrases)}*")
            time.sleep(0.2)

            # Check if this is an image-related prompt and we have an uploaded image
            if st.session_state.get('uploaded_image') and any(keyword in prompt.lower() for keyword in [
                "extract text", "get text", "ocr", "read text", "what text", "text from image",
                "correct", "fix", "debug", "analyze code", "review code", "improve code",
                "run", "execute", "output", "show output", "what is the output",
                "analyze", "describe", "explain", "what do you see", "what is in",
                "example", "similar", "write", "code", "topic"
            ]):
                # Process image-related prompt
                response = process_image_prompt(prompt, st.session_state.uploaded_image)
            else:
                # Regular prompt processing
                response = ask_ollama_stream(prompt, message_placeholder)

            # Ensure the placeholder shows the final response without caret
            message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save the updated chat with assistant's response
    save_current_chat()
    
    # Auto-scroll to latest message
    st.rerun()