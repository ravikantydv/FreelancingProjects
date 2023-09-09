import src.database as db
import src.setup

# Start implementing your task as mentioned in the problem statement


def insert_orders():
    insert_query1 = """INSERT INTO ecommerce_record.orders (order_id,customer_id,vendor_id,total_value,
            order_quantity, reward_point) values (101, 7, 2, 53985, 1, 100)"""
    db.create_insert_query(connection, insert_query1)
    insert_query2 = """INSERT INTO ecommerce_record.orders (order_id,customer_id,vendor_id,total_value,
            order_quantity, reward_point) values (102, 15, 5, 103758, 3, 300)"""
    db.create_insert_query(connection, insert_query2)
    insert_query3 = """INSERT INTO ecommerce_record.orders (order_id,customer_id,vendor_id,total_value,
            order_quantity, reward_point) values (103, 10, 1, 85537, 1, 200)"""
    db.create_insert_query(connection, insert_query3)
    insert_query4 = """INSERT INTO ecommerce_record.orders (order_id,customer_id,vendor_id,total_value,
            order_quantity, reward_point) values (104, 9, 3, 117684, 2, 300)"""
    db.create_insert_query(connection, insert_query4)
    insert_query5 = """INSERT INTO ecommerce_record.orders (order_id,customer_id,vendor_id,total_value,
            order_quantity, reward_point) values (105, 12, 3, 46743, 1, 100)"""
    db.create_insert_query(connection, insert_query5)
    print("5 new orders added")


def print_orders():
    query = "SELECT * FROM orders"
    list_of_orders = db.select_query(connection, query)
    print()
    print("List of orders:")
    print("{:<10} {:<12} {:<10} {:<12} {:<15} {:<15}".format('order_id', 'customer_id', 'vendor_id',
                                                             'total_value', 'order_quantity', 'reward_point'))

    for row in list_of_orders:
        order_id, total_value, order_quantity, reward_point, vendor_id, customer_id = row
        print("{:<10} {:<12} {:<10} {:<12} {:<15} {:<15}".format(order_id, customer_id, vendor_id,
                                                                 total_value, order_quantity, reward_point))
    print()


def max_and_min_orders():
    # Fetch max and min values from orders
    query_max = "SELECT MAX(total_value) from orders"
    res1 = db.select_query(connection, query_max)
    max_value = res1[0][0]
    query_min = "SELECT MIN(total_value) from orders"
    res2 = db.select_query(connection, query_min)
    min_value = res2[0][0]

    print(f"Maximum order value is {max_value} and minimum order value is {min_value}")
    print()


def value_greater_than_avg():
    query_avg = "SELECT AVG(total_value) FROM orders"
    res = db.select_query(connection, query_avg)
    avg = res[0][0]
    print(f"The average of order values is {avg:.2f}")
    query_list = "SELECT * FROM orders WHERE total_value>(SELECT AVG(total_value) FROM orders)"
    list_of_orders = db.select_query(connection, query_list)
    print("List of orders having value greater than average:")
    print("{:<10} {:<12} {:<10} {:<12} {:<15} {:<15}".format('order_id', 'customer_id', 'vendor_id',
                                                             'total_value', 'order_quantity', 'reward_point'))

    for row in list_of_orders:
        order_id, total_value, order_quantity, reward_point, vendor_id, customer_id = row
        print("{:<10} {:<12} {:<10} {:<12} {:<15} {:<15}".format(order_id, customer_id, vendor_id,
                                                                 total_value, order_quantity, reward_point))
    print()


def create_leaderboard():
    sql_create = """create table customer_leaderboard(
                customer_id varchar(10) not null, total_value double not null,
                customer_name varchar(50) not null, customer_email varchar(50) not null,
                foreign key (customer_id) references users(user_id), primary key (customer_id));"""
    db.create_table(connection, sql_create)

    sql_insert = """INSERT INTO customer_leaderboard(customer_id, total_value, customer_name, customer_email) 
                        SELECT orders.customer_id, max(orders.total_value), users.user_name, users.user_email 
                        FROM orders LEFT JOIN users ON orders.customer_id = users.user_id 
                        GROUP BY orders.customer_id ORDER BY orders.customer_id"""
    db.create_insert_query(connection, sql_insert)

    sql_select = "SELECT * FROM customer_leaderboard ORDER BY total_value DESC"
    leaderboard = db.select_query(connection, sql_select)
    print()
    print("Leaderboard:")
    print("{:<12} {:<15} {:<15} {:<20}".format('customer_id', 'total_value', 'customer_name', 'customer_email'))

    for row in leaderboard:
        customer_id, total_value, customer_name, customer_email = row
        print("{:<12} {:<15} {:<15} {:<20}".format(customer_id, total_value, customer_name, customer_email))
    print()


# Driver code
if __name__ == "__main__":
    """
    Please enter the necessary information related to the DB at this place. 
    Please change PW and ROOT based on the configuration of your own system. 
    """
    PW = "mysql"  # IMPORTANT! Put your MySQL Terminal password here.
    ROOT = "root"
    DB = "ecommerce_record"  # This is the name of the database we will create in the next step - call it whatever
    # you like.
    LOCALHOST = "localhost"
    connection = db.create_server_connection(LOCALHOST, ROOT, PW)

    # creating the schema in the DB
    db.create_and_switch_database(connection, DB, DB)

    # connect to the database
    connection = db.create_db_connection(LOCALHOST, ROOT, PW, DB)

    # Implement all the test cases and test them by running this file
    insert_orders()
    print_orders()
    max_and_min_orders()
    value_greater_than_avg()
    create_leaderboard()

    # close database connection
    db.close_db_connection(connection)