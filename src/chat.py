import os
import pandas as pd
import json
from datetime import datetime
import re
import time
from typing import Dict, List, Any

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
# Database imports
import sqlite3
from database_creation import create_db_and_tables
#logging imports
from app_logging import logger


# Token usage logger
class TokenUsageLogger:
    def __init__(self, log_file="../logs/token_usage.log"):
        self.log_file = log_file
        
        # Create or check log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,model_used,prompt_tokens,completion_tokens,total_tokens,query,response_length\n")
    
    def log_usage(self, model_used, prompt_tokens, completion_tokens, total_tokens, query, response_length):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sanitize query for CSV
        query = query.replace(",", " ").replace("\n", " ")
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{model_used},{prompt_tokens},{completion_tokens},{total_tokens},{query},{response_length}\n")
        
        logger.info(f"Token usage - Model: {model_used}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

# Initialize database
create_db_and_tables()

# function to check wether the customer exist or not
def get_customer_by_email(email):
    """Retrieve customer data by email"""
    logger.info(f"Searching for customer with email: {email}")
    start_time = time.time()
    
    conn = sqlite3.connect('../ecommerce_support.db')
    query = "SELECT * FROM customers WHERE email = ?"
    customer = pd.read_sql_query(query, conn, params=(email,))
    conn.close()
    
    result = customer.to_dict('records')[0] if not customer.empty else None
    
    elapsed_time = time.time() - start_time
    if result:
        logger.info(f"Found customer: {result.get('first_name', '')} {result.get('last_name', '')} in {elapsed_time:.2f} seconds")
    else:
        logger.warning(f"No customer found with email: {email} (search took {elapsed_time:.2f} seconds)")
    
    return result

# function to fetch orders of particular customer
def get_customer_orders(customer_id):
    """Retrieve order history for a customer"""
    logger.info(f"Retrieving orders for customer_id: {customer_id}")
    start_time = time.time()
    
    conn = sqlite3.connect('../ecommerce_support.db')
    query = "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date DESC"
    orders = pd.read_sql_query(query, conn, params=(customer_id,))
    conn.close()
    
    # Parse items_json
    if not orders.empty:
        for i, row in orders.iterrows():
            try:
                orders.at[i, 'items'] = json.loads(row['items_json'])
            except Exception as e:
                logger.error(f"Error parsing items_json for order {row.get('order_id', 'unknown')}: {str(e)}")
                orders.at[i, 'items'] = []
    
    result = orders.to_dict('records')
    
    elapsed_time = time.time() - start_time
    logger.info(f"Retrieved {len(result)} orders for customer {customer_id} in {elapsed_time:.2f} seconds")
    
    return result

#function to fetch the order details 
def get_order_details(order_id):
    """Retrieve detailed information about an order"""
    logger.info(f"Retrieving details for order: {order_id}")
    start_time = time.time()
    
    conn = sqlite3.connect('../ecommerce_support.db')
    query = """
    SELECT o.*, c.first_name, c.last_name, c.email
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_id = ?
    """
    order = pd.read_sql_query(query, conn, params=(order_id,))
    
    if not order.empty:
        order_data = order.to_dict('records')[0]
        
        # Parse items and add product details
        try:
            items = json.loads(order_data['items_json'])
            order_data['items'] = []
            
            for item in items:
                product_id = item['product_id']
                query = "SELECT * FROM products WHERE product_id = ?"
                product = pd.read_sql_query(query, conn, params=(product_id,))
                
                if not product.empty:
                    product_data = product.to_dict('records')[0]
                    item['product_name'] = product_data['name']
                    item['product_category'] = product_data['category']
                    item['return_policy'] = product_data['return_policy']
                    
                order_data['items'].append(item)
                
        except Exception as e:
            logger.error(f"Error parsing order items for order {order_id}: {str(e)}")
            order_data['items'] = []
            
        conn.close()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Retrieved order details for {order_id} with {len(order_data.get('items', []))} items in {elapsed_time:.2f} seconds")
        
        return order_data
    
    conn.close()
    
    elapsed_time = time.time() - start_time
    logger.warning(f"No order found with ID: {order_id} (search took {elapsed_time:.2f} seconds)")
    
    return None


# function to check the product details
def get_product_details(product_id):
    """Retrieve product information"""
    logger.info(f"Retrieving product details for product_id: {product_id}")
    start_time = time.time()
    
    conn = sqlite3.connect('../ecommerce_support.db')
    query = "SELECT * FROM products WHERE product_id = ?"
    product = pd.read_sql_query(query, conn, params=(product_id,))
    conn.close()
    
    result = product.to_dict('records')[0] if not product.empty else None
    
    elapsed_time = time.time() - start_time
    if result:
        logger.info(f"Found product: {result.get('name', '')} in {elapsed_time:.2f} seconds")
    else:
        logger.warning(f"No product found with ID: {product_id} (search took {elapsed_time:.2f} seconds)")
    
    return result


# function to check the return policy for particular category
def get_return_policy_for_category(category):
    """Get return policy for a product category"""
    logger.info(f"Retrieving return policy for category: {category}")
    start_time = time.time()
    
    conn = sqlite3.connect('../ecommerce_support.db')
    query = "SELECT DISTINCT return_policy FROM products WHERE category = ?"
    result = pd.read_sql_query(query, conn, params=(category,))
    conn.close()
    
    policy = result['return_policy'].iloc[0] if not result.empty else None
    
    elapsed_time = time.time() - start_time
    if policy:
        logger.info(f"Found return policy for category {category} in {elapsed_time:.2f} seconds")
    else:
        logger.warning(f"No return policy found for category: {category} (search took {elapsed_time:.2f} seconds)")
    
    return policy


# RAG System Setup
def create_vector_store():
    """Create vector store from FAQs"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(current_dir, "chroma_db")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
        logger.info("Vector store already exists. Skipping creation.")
        # or load the existing store if needed
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        return vectorstore
    logger.info("Creating vector store from FAQs and product information")
    start_time = time.time()
    
    # Load FAQs
    try:
        faqs_df = pd.read_csv(r"C:\Users\HP\ECommerce_AI\Ecom_Chatbot\data\faqs.csv")
        logger.info(f"Loaded {len(faqs_df)} FAQs")
        
        # Create documents for vector store
        documents = []
        for _, row in faqs_df.iterrows():
            content = f"Question: {row['question']}\nAnswer: {row['answer']}\nCategory: {row['category']}"
            metadata = {"id": row['faq_id'], "category": row['category']}
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Add product information
        conn = sqlite3.connect('../ecommerce_support.db')
        products_df = pd.read_sql_query("SELECT * FROM products", conn)
        conn.close()
        
        logger.info(f"Adding {len(products_df)} products to vector store")
        
        for _, row in products_df.iterrows():
            content = f"Product ID: {row['product_id']}\nName: {row['name']}\n"
            content += f"Category: {row['category']}\nPrice: ${row['price']}\n"
            content += f"Description: {row['description']}\n"
            content += f"Return Policy: {row['return_policy']}\n"
            content += f"Stock Quantity: {row['stock_quantity']}"
            
            metadata = {"id": row['product_id'], "type": "product"}
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Created {len(split_docs)} document chunks for embedding")
        
        # Create embeddings
        #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Vector store created successfully in {elapsed_time:.2f} seconds")
        
        return vectorstore
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error creating vector store after {elapsed_time:.2f} seconds: {str(e)}")
        raise

# Conversation Context Manager
class ConversationState:
    def __init__(self):
        self.last_topic = None
        self.pending_questions = []
        self.escalation_attempts = 0
        self.unresolved_issues = []
        self.previous_query = None
        self.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S") + str(id(self))[-6:]
        
        logger.info(f"Initialized new conversation with ID: {self.conversation_id}")




# RAG Chain Setup
def setup_qa_chain(vectorstore):
    """Set up RAG chain with conversational memory"""
    logger.info("Setting up QA chain with RAG and conversational memory")
    start_time = time.time()
    
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create system prompt
        system_prompt = """You are an AI customer support agent for an e-commerce platform.
        Your name is ShopAssist, and you help customers with their questions about orders, products, returns, and general inquiries.
        
        When responding to customer queries:
        1. Be professional, helpful, and friendly
        2. Use the retrieved context to provide accurate information
        3. If you don't know the answer, be honest and offer to escalate the issue to a human agent
        4. For specific order inquiries, ask for order number or customer email if not provided
        5. Format currency values with $ symbol and two decimal places
        6. Keep responses concise but complete
        7. Remember to refer to previous conversation context when continuing a discussion
        8.  If you are displaying order details, ensure that the information is presented in a clear and structured format with each detail on a new line. For example:
            - Order ID: O10026
            - Date: 2024-02-05
            - Status: Delivered
            - Tracking Number: TRK789012370
            - Estimated Delivery: 2024-02-12
            - Total: $29.99
            - Shipping Method: Standard
        9. If the the question is like "can i cancel it?" answer yes it is possible.
       10. If asked about a product or reviews guide the user by mentioning the steps to do that task in website.
        Context information: {context}
        
        Customer's question: {question}
        
        Conversation history: {chat_history}
        """
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=system_prompt,
        )
        
        # Set up memory
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
        
        # Create LLM
        model_name = "models/gemini-1.5-pro"
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        logger.info(f"Using LLM model: {model_name}")
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory,
            }
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"QA chain setup completed in {elapsed_time:.2f} seconds")
        
        return qa_chain
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error setting up QA chain after {elapsed_time:.2f} seconds: {str(e)}")
        raise
    

# Customer Context Manager
class CustomerContext:
    def __init__(self):
        self.current_customer = None
        self.current_email = None
        self.current_order = None
        self.orders = []
        self.product_category = None
        self.recent_returns = []
        
        logger.info("Initialized new CustomerContext")
    
    def set_customer_by_email(self, email):
        """Set current customer by email and fetch their orders"""
        logger.info(f"Setting customer context by email: {email}")
        self.current_email = email
        self.current_customer = get_customer_by_email(email)
        
        # Fetch all orders for this customer
        if self.current_customer:
            customer_id = self.current_customer['customer_id']
            self.orders = get_customer_orders(customer_id)
            logger.info(f"Retrieved {len(self.orders)} orders for customer ID {customer_id}")
            
            # Set the most recent order as current_order if available
            if self.orders:
                self.current_order = self.orders[0]
                logger.info(f"Set current order to most recent: {self.current_order.get('order_id', 'unknown')}")
        else:
            logger.warning(f"No customer found with email: {email}")
                
        return self.current_customer
    
    def set_current_order(self, order_id):
        """Set current order by order ID"""
        logger.info(f"Setting current order to: {order_id}")
        self.current_order = get_order_details(order_id) ## fetches order details
        
        if self.current_order:
            logger.info(f"Current order successfully set to {order_id}")
        else:
            logger.warning(f"Failed to set current order: no order found with ID {order_id}")
            
        return self.current_order
    
    def get_customer_context(self):
        """Get context information about the current customer"""
        if not self.current_customer:
            logger.info("No customer context available")
            return "No customer information available."
        
        customer = self.current_customer
        
        context = f"Customer: {customer['first_name']} {customer['last_name']}\n"
        context += f"Email: {customer['email']}\n"
        context += f"Loyalty Tier: {customer['loyalty_tier']}\n"
        context += f"Customer since: {customer['joined_date']}\n\n"
        
        if self.orders:
            context += f"Order History ({len(self.orders)} orders):\n"
            for order in self.orders[:3]:  # Just include the 3 most recent orders
                context += f"- Order {order['order_id']} ({order['order_date']}): ${order['total_amount']:.2f} - Status: {order['status']}\n"
        
        logger.info(f"Generated customer context for {customer['first_name']} {customer['last_name']}")
        return context
    
    def get_order_summary(self):
        """Get a summary of the customer's orders"""
        if not self.orders:
            logger.info("No orders found for customer")
            return "No orders found for this customer."
        
        summary = f"Order Summary for {self.current_customer['first_name']} {self.current_customer['last_name']}:\n\n"
        
        for order in self.orders:
            summary += f"Order ID: {order['order_id']}\n"
            summary += f"Date: {order['order_date']}\n"
            summary += f"Status: {order['status']}\n"
            summary += f"Total: ${order['total_amount']:.2f}\n"
            summary += f"Shipping Method: {order['shipping_method']}\n"
            
            if order['tracking_number']:
                summary += f"Tracking Number: {order['tracking_number']}\n"
                
            if order['estimated_delivery']:
                summary += f"Estimated Delivery: {order['estimated_delivery']}\n"
                
            summary += "\n"
        
        logger.info(f"Generated order summary for customer with {len(self.orders)} orders")
        return summary
    
    def get_current_order_details(self):
        """Get details about the current order"""
        if not self.current_order:
            logger.info("No current order selected")
            return "No current order selected."
        
        order = self.current_order
        details = f"Order Details for {order['order_id']}:\n\n"
        details += f"Date: {order['order_date']}\n"
        details += f"Status: {order['status']}\n"
        details += f"Total: ${order['total_amount']:.2f}\n"
        details += f"Shipping Method: {order['shipping_method']}\n"
        
        if order['tracking_number']:
            details += f"Tracking Number: {order['tracking_number']}\n"
            
        if order['estimated_delivery']:
            details += f"Estimated Delivery: {order['estimated_delivery']}\n"
            
        if 'items' in order and order['items']:
            details += "\nItems:\n"
            for item in order['items']:
                details += f"- {item.get('quantity', 1)}x {item.get('product_name', 'Unknown Product')} "
                details += f"(${float(item.get('price', 0)):.2f} each)\n"
                if 'return_policy' in item:
                    details += f"  Return Policy: {item.get('return_policy')}\n"
        
        logger.info(f"Generated order details for order {order['order_id']}")
        return details
    
    def get_return_info_for_current_order(self):
        """Get return eligibility information for current order"""
        if not self.current_order:
            logger.info("Cannot get return info: no current order selected")
            return "No order selected. Please provide an order number to check return eligibility."
        
        order = self.current_order
        order_date = datetime.strptime(order['order_date'], '%Y-%m-%d')
        today = datetime.now()
        days_since_order = (today - order_date).days
        
        info = f"Return Information for Order {order['order_id']}:\n\n"
        info += f"Order Date: {order['order_date']} ({days_since_order} days ago)\n"
        info += f"Status: {order['status']}\n\n"
        
        if 'items' in order and order['items']:
            info += "Return Eligibility by Item:\n"
            for item in order['items']:
                product_name = item.get('product_name', 'Unknown Product')
                return_policy = item.get('return_policy', 'Standard 30-day return policy')
                
                # Extract days from return policy
                days_match = re.search(r'(\d+)[-\s]?day', return_policy.lower())
                return_window = int(days_match.group(1)) if days_match else 30
                
                eligible = days_since_order <= return_window
                
                info += f"- {product_name}: "
                if eligible:
                    info += f"ELIGIBLE (within {return_window}-day window)\n"
                    info += f"  Return window ends in {return_window - days_since_order} days\n"
                else:
                    info += f"NOT ELIGIBLE (outside {return_window}-day window)\n"
                    info += f"  Return window expired {days_since_order - return_window} days ago\n"
        
        logger.info(f"Generated return information for order {order['order_id']}")
        return info

# Chatbot Interface
class EcommerceChatbot:
    def __init__(self):
        logger.info("Initializing EcommerceChatbot")
        start_time = time.time()
        
        # Create database
        create_db_and_tables()
        
        # Set up vector store and QA chain
        self.vectorstore = create_vector_store()
        self.qa_chain = setup_qa_chain(self.vectorstore)
        
        # Initialize customer context
        self.customer_context = CustomerContext()
        
        # Initialize conversation state
        self.conversation_state = ConversationState()
        
        # Initialize token usage logger
        self.token_logger = TokenUsageLogger()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Chatbot initialized successfully in {elapsed_time:.2f} seconds!")
    
    # Complete the EcommerceChatbot class's process_query method
    def process_query(self,query, email=None, order_id=None):
        """Process customer query with context"""
        query_start_time = time.time()
        logger.info(f"Processing query: '{query}' (email: {email}, order_id: {order_id})")
        
        # Save the previous query for context
        prev_query = self.conversation_state.previous_query
        self.conversation_state.previous_query = query
        
        # Update customer context if email provided
        customer_updated = False
        if email and email != self.customer_context.current_email:
            customer = self.customer_context.set_customer_by_email(email) ## checking for valid customer
            customer_updated = True
            if not customer:
                response = f"No customer found with email: {email}"
                query_time = time.time() - query_start_time
                logger.info(f"Query processed in {query_time:.2f} seconds with response: {response}")
                return response
        
        # Update order context if order_id provided
        if order_id:
            order = self.customer_context.set_current_order(order_id) ## fetches order details
            if not order:
                response = f"No order found with ID: {order_id}"
                query_time = time.time() - query_start_time
                logger.info(f"Query processed in {query_time:.2f} seconds with response: {response}")
                return response
        
        # Get customer context
        context_info = self.customer_context.get_customer_context()  # fetches customer details
        
        # Check for topic patterns
        is_order_query = any(term in query.lower() for term in ["order","orders"])# "status", "tracking", "delivery", "package", "shipment"
        is_return_query = any(term in query.lower() for term in ["return", "refund", "send back", "money back", "exchange"])
        
        # Check for topic continuity with the previous query
        previous_context = ""
        if prev_query:
            previous_context = f"Prior query: {prev_query}\n"
            
            # If previous query was about returns and current query contains email/order info
            if any(term in prev_query.lower() for term in ["return", "refund"]) and (email or '@' in query or order_id or query.strip().lower() == "yes"):
                is_return_query = True
        
        # Handle direct requests for order information
        if is_order_query:
            logger.info("Query classified as order-related")
            # If we just updated the customer and this is an order query, return order info directly
            if customer_updated or "email" in query.lower():
                if self.customer_context.orders:
                    response = self.customer_context.get_order_summary()  ## all order history
                    query_time = time.time() - query_start_time
                    logger.info(f"Order query processed in {query_time:.2f} seconds")
                    return response
                else:
                    response = f"I don't see any orders associated with {email}. If you've placed an order recently, it may not have been processed yet."
                    query_time = time.time() - query_start_time
                    logger.info(f"Order query processed in {query_time:.2f} seconds (no orders found)")
                    return response
            
            # If we have a specific current order
            if self.customer_context.current_order:
                response = self.customer_context.get_current_order_details()
                query_time = time.time() - query_start_time
                logger.info(f"Order query processed in {query_time:.2f} seconds")
                return response
            
            # If we have orders but no specific one selected
            if self.customer_context.orders:
                response = self.customer_context.get_order_summary()
                query_time = time.time() - query_start_time
                logger.info(f"Order query processed in {query_time:.2f} seconds")
                return response
            
            # If we have customer info but no orders
            if self.customer_context.current_customer:
                response = f"I don't see any orders associated with your account. If you've placed an order recently, it may not have been processed yet."
                query_time = time.time() - query_start_time
                logger.info(f"Order query processed in {query_time:.2f} seconds (no orders found)")
                return response
        
        # Handle return requests
        if is_return_query:
            logger.info("Query classified as return-related")
            # Customer provided email after asking about returns
            if customer_updated and prev_query and "return" in prev_query.lower():
                if self.customer_context.orders:
                    response = self.customer_context.get_return_info_for_current_order()
                    query_time = time.time() - query_start_time
                    logger.info(f"Return query processed in {query_time:.2f} seconds")
                    return response
                else:
                    response = f"I don't see any recent orders associated with {email} that would be eligible for return. If you've placed an order recently and have the order number, please provide it."
                    query_time = time.time() - query_start_time
                    logger.info(f"Return query processed in {query_time:.2f} seconds (no orders found)")
                    return response
            
            # If we have a current order
            if self.customer_context.current_order:
                response = self.customer_context.get_return_info_for_current_order()
                query_time = time.time() - query_start_time
                logger.info(f"Return query processed in {query_time:.2f} seconds")
                return response
                
            # If previous query contained frustration indicators and this is a follow-up about returns
            frustration_indicators = ["not working", "help me", "no", "can't", "still", "again", "unsatisfied", "unhappy", "angry", "upset"]
            if any(indicator in query.lower() for indicator in frustration_indicators) or self.conversation_state.escalation_attempts > 0:
                self.conversation_state.escalation_attempts += 1
                if self.conversation_state.escalation_attempts >= 2:
                    response = "I understand this return issue is important to you. Let me connect you with a human customer service representative who can better assist with your specific situation. They'll reach out to you shortly. In the meantime, could you please provide any order details you have to help them assist you faster?"
                    query_time = time.time() - query_start_time
                    logger.info(f"Return query escalated in {query_time:.2f} seconds")
                    return response
        
        # Add relevant context based on query type
        enhanced_query = query
        if is_order_query and self.customer_context.current_order:
            order_details = f"""Current Order: {self.customer_context.current_order['order_id']}\n
            Date: {self.customer_context.current_order['order_date']}\n
            Status: {self.customer_context.current_order['status']}\n
            Tracking (if shipped): {self.customer_context.current_order['tracking_number'] or 'N/A'}
            Estimated Delivery: {self.customer_context.current_order['estimated_delivery'] or 'N/A'}
            """
            enhanced_query = f"{order_details}\n\nCustomer question about this order: {query}"
        
        # Use RAG to answer the query
        logger.info("Passing query to RAG system")
        full_context = f"{previous_context}{context_info}"
        
        try:
            with get_openai_callback() as cb:
                start_rag_time = time.time()
                result = self.qa_chain.invoke({"query": enhanced_query, "context": full_context})
                rag_time = time.time() - start_rag_time
                
                # Log token usage
                self.token_logger.log_usage(
                    model_used="gemini-1.5-pro",
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens,
                    total_tokens=cb.total_tokens,
                    query=query,
                    response_length=len(result["result"]) if "result" in result else 0
                )
                
                logger.info(f"RAG query processed in {rag_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            response = "I apologize, but I'm experiencing technical difficulties. Please try again or contact customer support directly."
            query_time = time.time() - query_start_time
            logger.error(f"Query failed after {query_time:.2f} seconds")
            return response
        
        response = result["result"] if "result" in result else "I apologize, but I couldn't find an answer to your question."
        
        # Extract and look for email addresses in query to update customer context
        if not email:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_matches = re.findall(email_pattern, query)
            if email_matches:
                potential_email = email_matches[0]
                logger.info(f"Found potential email in query: {potential_email}")
                customer = self.customer_context.set_customer_by_email(potential_email)
                if customer:
                    response += f"\n\nI've found your customer information, {customer['first_name']}. Is there anything else you'd like to know about your orders or account?"
        
        # Extract and look for order IDs in query
        if not order_id:
            # Assuming order IDs are alphanumeric with possible hyphens, 5-12 characters
            order_pattern = r'\b[A-Za-z0-9]{3,5}-?[A-Za-z0-9]{2,7}\b'
            order_matches = re.findall(order_pattern, query)
            for potential_order in order_matches:
                logger.info(f"Found potential order ID in query: {potential_order}")
                order = self.customer_context.set_current_order(potential_order)
                if order:
                    response += f"\n\nI've found your order {potential_order}. Let me know if you need more details about this order."
                    break
        
        query_time = time.time() - query_start_time
        logger.info(f"Total query processing time: {query_time:.2f} seconds")
        return response