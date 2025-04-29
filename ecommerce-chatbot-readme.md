# E-Commerce Chatbot

An intelligent AI-powered customer support chatbot for e-commerce platforms that helps customers with their orders, returns, product inquiries, and more.

## 📋 Overview

This project implements a sophisticated customer support chatbot using RAG (Retrieval Augmented Generation) technology powered by Google's Generative AI models. The chatbot can handle customer queries about orders, returns, product information, and general FAQs, providing personalized responses based on customer data.

## ✨ Features

- **Customer Information Retrieval**: Identifies customers by email and retrieves their information.
- **Order Management**: Retrieves order history and provides detailed information about specific orders.
- **Return Processing**: Checks return eligibility for products and provides return instructions.
- **Product Information**: Provides details about products including price, description, and availability.
- **Context-Aware Conversations**: Maintains conversation state to provide contextually relevant responses.
- **RAG-Based Knowledge Retrieval**: Uses vector embeddings to retrieve relevant information from FAQs and product data.
- **Token Usage Logging**: Tracks token usage for monitoring and optimization.

## 🛠️ Technical Architecture

The chatbot is built with:
- **LangChain**: For RAG implementation and conversation management
- **Google Generative AI**: For embeddings and text generation (Gemini models)
- **SQLite**: For storing customer, order, and product information
- **Chroma DB**: For vector storage and retrieval
- **Pandas**: For data manipulation

## 📁 Project Structure

```
├── chat.py              # Main chatbot implementation
├── database_creation.py # Database setup script
├── app_logging.py       # Logging configuration
├── data/
│   └── faqs.csv         # FAQ data for vector store
├── logs/
│   └── token_usage.log  # Token usage tracking
└── ecommerce_support.db # SQLite database
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google API key for Generative AI access

### Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/ecommerce-chatbot.git
   cd ecommerce-chatbot
   ```

2. **Install dependencies**
   ```
   pip install langchain langchain-google-genai pandas sqlite3 chromadb
   ```

3. **Set up environment variables**
   ```
   export GOOGLE_API_KEY=your_api_key_here
   ```

4. **Initialize the database**
   ```
   python database_creation.py
   ```

5. **Run the chatbot**
   ```
   python main.py  # Create this file to instantiate and run the EcommerceChatbot class
   ```

## 💬 Usage Examples

```python
# Initialize the chatbot
chatbot = EcommerceChatbot()

# Process a query without customer context
response = chatbot.process_query("What is your return policy?")
print(response)

# Process a query with customer email
response = chatbot.process_query(
    "Show me my recent orders", 
    email="customer@example.com"
)
print(response)

# Process a query with specific order
response = chatbot.process_query(
    "Can I return the items in this order?", 
    order_id="O10026"
)
print(response)
```

## 📊 Data Model

The system uses the following database tables:
- `customers`: Customer information and loyalty status
- `orders`: Order history with shipping and tracking information
- `products`: Product catalog with descriptions and return policies

## 🔧 Configuration

You can customize the following aspects:
- Vector store chunk size and overlap in `create_vector_store()`
- System prompt for the AI in `setup_qa_chain()`
- Token usage logging path in `TokenUsageLogger`

## 🛡️ Error Handling

The system includes robust error handling for:
- Database connection issues
- LLM token limits and API failures
- Customer or order not found scenarios
- JSON parsing of order items

## 📈 Future Improvements

- Add web interface for direct customer interaction
- Implement authentication for security
- Add support for multimedia responses (images, videos)
- Integrate with payment processing systems
- Add multi-language support

## 📄 License

[MIT License](LICENSE)

## 👥 Contributors

- Your Name - Initial work

---

Feel free to open issues or submit pull requests with improvements!
