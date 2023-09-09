import csv
import src.database as db

PW = "mysql"  # IMPORTANT! Put your MySQL Terminal password here.
ROOT = "root"
DB = "ecommerce_record"  # This is the name of the database we will create in the next step - call it whatever you like.
LOCALHOST = "localhost"  # considering you have installed MySQL server on your computer

RELATIVE_CONFIG_PATH = '../config/'

USER = 'users'
PRODUCTS = 'products'
ORDER = 'orders'

connection = db.create_server_connection(LOCALHOST, ROOT, PW)

# creating the schema in the DB
db.create_and_switch_database(connection, DB, DB)

# Create the tables through python code here
# if you have created the table in UI, then no need to define the table structure
# If you are using python to create the tables, call the relevant query to complete the creation
connection = db.create_db_connection(LOCALHOST, ROOT, PW, DB)

with open(RELATIVE_CONFIG_PATH + USER + '.csv', 'r') as f:
    val = []
    data = csv.reader(f)
    for row in data:
        val.append(tuple(row))
    val.pop(0)
    """
    Here we have accessed the file data and saved into the val data structure, which list of tuples. 
    Now you should call appropriate method to perform the insert operation in the database. 
    """
    sql = """INSERT INTO ecommerce_record.users (user_id, user_name, user_email, user_password, user_address,
        is_vendor) values (%s, %s, %s, %s, %s, %s)"""
    db.insert_many_records(connection, sql, val)

with open(RELATIVE_CONFIG_PATH + PRODUCTS + '.csv', 'r') as f:
    val = []
    data = csv.reader(f)
    for row in data:
        val.append(tuple(row))
    val.pop(0)
    """
    Here we have accessed the file data and saved into the val data structure, which list of tuples. 
    Now you should call appropriate method to perform the insert operation in the database. 
    """
    sql = """INSERT INTO ecommerce_record.products (product_id,product_name,product_price,product_description,
        vendor_id,emi_available) values (%s, %s, %s, %s, %s, %s)"""
    db.insert_many_records(connection, sql, val)

with open(RELATIVE_CONFIG_PATH + ORDER + '.csv', 'r') as f:
    val = []
    data = csv.reader(f)
    for row in data:
        val.append(tuple(row))

    val.pop(0)
    """
    Here we have accessed the file data and saved into the val data structure, which list of tuples. 
    Now you should call appropriate method to perform the insert operation in the database. 
    """
    sql = """INSERT INTO ecommerce_record.orders (order_id,customer_id,vendor_id,total_value,order_quantity,
        reward_point) values (%s, %s, %s, %s, %s, %s)"""
    db.insert_many_records(connection, sql, val)

db.close_db_connection(connection)