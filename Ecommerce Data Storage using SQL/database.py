import mysql.connector
import random
import time
import datetime


# Global methods to push interact with the Database

# This method establishes the connection with the MySQL
def create_server_connection(host_name, user_name, user_password):
    # Implement the logic to create the server connection
    try:
        connection = mysql.connector.connect(host=host_name, user=user_name, password=user_password)
    except mysql.connector.Error as err:
        print(err)
    else:
        if connection.is_connected():
            print("Connected to MySQL server")
            return connection


# This method will create the database and make it an active database
def create_and_switch_database(connection, db_name, switch_db):
    # For database creation use this method
    # If you have created your database using UI, no need to implement anything
    pass


# This method will establish the connection with the newly created DB
def create_db_connection(host_name, user_name, user_password, db_name):
    try:
        connection = mysql.connector.connect(host=host_name, user=user_name, password=user_password,
                                             database=db_name)
    except mysql.connector.Error as err:
        print(err)
    else:
        if connection.is_connected():
            print("Connected to database")
            return connection


# Use this function to create the tables in a database
def create_table(connection, table_creation_statement):
    # If you have created your tables using UI, no need to implement anything
    cursor = connection.cursor(prepared=True)
    cursor.execute(table_creation_statement)
    cursor.close()


# Perform all single insert statements in the specific table through a single function call
def create_insert_query(connection, query):
    # This method will perform creation of the table
    # this can also be used to perform single data point insertion in the desired table
    if not query:
        return None
    cursor = connection.cursor(prepared=True)
    cursor.execute(query)
    connection.commit()
    cursor.close()


# retrieving the data from the table based on the given query
def select_query(connection, query):
    # fetching the data points from the table
    cursor = connection.cursor(prepared=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result


# Execute multiple insert statements in a table
def insert_many_records(connection, sql, val):
    cursor = connection.cursor(prepared=True)
    cursor.executemany(sql, val)
    connection.commit()
    cursor.close()


def close_db_connection(connection):
    if connection.is_connected():
        connection.close()
        print("DB connection closed")