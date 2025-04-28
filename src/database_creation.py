import os
import sqlite3
import pandas as pd
from app_logging import logger
import time

def create_db_and_tables():
    """Create SQLite database and load data"""
    
    if os.path.exists('../ecommerce_support.db'):
          logger.info("Database already exists. Skipping creation and data load.")
          return
    
    start_time = time.time()
    conn = sqlite3.connect('../ecommerce_support.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
    CREATE TABLE IF NOT EXISTS customers
    (customer_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, email TEXT, phone TEXT,
    address TEXT, city TEXT, state TEXT, zip_code TEXT, joined_date TEXT, loyalty_tier TEXT)
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS orders
    (order_id TEXT PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT, 
    total_amount REAL, shipping_method TEXT, tracking_number TEXT, estimated_delivery TEXT, items_json TEXT)
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS products
    (product_id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL, description TEXT,
    stock_quantity INTEGER, rating REAL, return_policy TEXT)
    ''')
    
    # Load data from CSV files
    try:
        customers_df = pd.read_csv('../data/customers.csv')
        orders_df = pd.read_csv('../data/orders.csv')
        products_df = pd.read_csv('../data/products.csv')
        
        # Insert data
        customers_df.to_sql('customers', conn, if_exists='replace', index=False)
        orders_df.to_sql('orders', conn, if_exists='replace', index=False)
        products_df.to_sql('products', conn, if_exists='replace', index=False)
        
        logger.info(f"Loaded {len(customers_df)} customers, {len(orders_df)} orders, and {len(products_df)} products")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        conn.close()
        raise
    
    conn.commit()
    conn.close()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Database created and populated successfully in {elapsed_time:.2f} seconds")