import streamlit as st
import os
import sys
import time
from chat import EcommerceChatbot
import traceback
from dotenv import load_dotenv
load_dotenv()
# Set page configuration
st.set_page_config(page_title="ShopAssist Customer Support", page_icon="🛒", layout="wide")

# Function to initialize the session state
def initialize_session_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'email' not in st.session_state:
        st.session_state.email = ""
    if 'order_id' not in st.session_state:
        st.session_state.order_id = ""
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False
    if 'connection_errors' not in st.session_state:
        st.session_state.connection_errors = 0

# Function to set up the environment
def setup_environment():
    """Set up the environment for the chatbot"""
    # Check if data files exist
    required_files = ['customers.csv', 'orders.csv', 'products.csv', 'faqs.csv']
    for file in required_files:
        data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
        file_path = os.path.join(data_folder, file)
        if not os.path.exists(file_path):
            st.error(f"Error: {file} not found. Please ensure all data files are in the directory.")
            return False
    
    # Set up Google API key
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    return True

# Initialize chatbot
def initialize_chatbot():
    with st.spinner('Initializing chatbot, please wait...'):
        try:
            st.session_state.chatbot = EcommerceChatbot()
            st.session_state.is_initialized = True
            st.session_state.connection_errors = 0
            st.success("Chatbot initialized successfully!")
            return True
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.session_state.is_initialized = False
            return False

# Process message with error handling
def process_message(user_message):
    if not st.session_state.is_initialized:
        st.warning("Chatbot is not initialized yet. Please wait.")
        return
    
    if user_message:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Extract email and order ID from sidebar if available
        email = st.session_state.email if st.session_state.email else None
        order_id = st.session_state.order_id if st.session_state.order_id else None
        
        # Process query through the chatbot with timeout handling
        with st.spinner('Processing your query...'):
            try:
                # Add a timeout to prevent extremely long waits
                start_time = time.time()
                timeout = 30  # 30 seconds timeout
                
                # Try to process the query
                response = st.session_state.chatbot.process_query(
                    user_message, 
                    email=email, 
                    order_id=order_id
                )
                
                # Reset connection error counter on success
                st.session_state.connection_errors = 0
                
            except Exception as e:
                error_msg = str(e)
                stack_trace = traceback.format_exc()
                
                # Check if it's a connection/timeout error
                if "timeout" in error_msg.lower() or "503" in error_msg or "connect" in error_msg.lower():
                    st.session_state.connection_errors += 1
                    
                    if st.session_state.connection_errors >= 3:
                        # After 3 consecutive connection errors, suggest reinitialization
                        response = ("I'm having trouble connecting to the service. This might be due to network issues "
                                   "or service unavailability. Please try reinitializing the chatbot from the sidebar.")
                        st.error("Multiple connection errors detected. Consider reinitializing the chatbot.")
                    else:
                        response = ("I'm having trouble connecting right now. This might be a temporary issue. "
                                   "Please try your question again in a moment.")
                else:
                    # Other type of error
                    response = f"I encountered an error while processing your query. Please try again with a different question."
        
        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Set up the environment
    if not setup_environment():
        st.stop()
    
    # Display header
    st.title("🛒 ShopAssist E-commerce Customer Support")
    
    # Create a sidebar for settings
    with st.sidebar:
        st.header("Customer Information")
        
        # Email input
        st.text_input("Email Address", key="email")
        
        # Order ID input
        st.text_input("Order ID", key="order_id")
        
        # Initialize/Reinitialize chatbot button
        button_text = "Initialize Chatbot" if not st.session_state.is_initialized else "Reinitialize Chatbot"
        if st.button(button_text):
            if initialize_chatbot():
                # Clear messages on reinitialization
                if st.session_state.messages:
                    st.session_state.messages = []
                    st.rerun()
        
        if st.session_state.is_initialized:
            st.success("Chatbot is ready!")
            
            # Add a connection status indicator
            if st.session_state.connection_errors > 0:
                st.warning(f"Connection issues detected: {st.session_state.connection_errors}")
        
        # Display some info
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("""
        1. Enter your email address (optional)
        2. Enter your order ID if you have one (optional)
        3. Ask questions about your orders, products, returns, etc.
        """)
        
        # Add offline mode option
        st.markdown("---")
        
    # Main chat area
    if not st.session_state.is_initialized:
        # Show welcome message if chatbot not initialized
        st.info("Please initialize the chatbot using the button in the sidebar to start.")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Add welcome message if this is the first interaction
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                welcome_msg = "Hello! I'm ShopAssist, your e-commerce support assistant. How can I help you today?"
                st.markdown(welcome_msg)
                st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        # User input
        user_input = st.chat_input("Type your message here...")
        
        # Process message if user has entered something
        if user_input:
            process_message(user_input)
            # Force a rerun to update the chat with the new messages
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()