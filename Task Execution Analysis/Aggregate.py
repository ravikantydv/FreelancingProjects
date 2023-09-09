from src.LinkedList import LinkedList # Importing library

"""
Aggregate class is responsible for implementating various operations on the linked list
    1. Get Minimum Timed Task Details
    2. Get Maximum Timed Task Details
    3. Get Average of all the execution times of the tasks pushed in the linked list
"""
class Aggregate: 
    
    #Initializing linked list object for various operations 
    def __init__(self, linked_list:LinkedList):
        self.linked_list = linked_list
    
    #Function responsible for searching the task having maximum execution time among all the tasks
    def get_maximised_time_task(self):
        temp = self.linked_list.head
        max_time = temp.end_time - temp.start_time
        max_task = None
        while temp:
            time = temp.end_time - temp.start_time
            if time >= max_time:
                max_time = time
                max_task = temp.task_id
            temp = temp.next
        return max_task, max_time
    
    #Function responsible for searching the task having minimum execution time among all the tasks
    def get_minimised_timed_task(self):
        temp = self.linked_list.head
        min_time = temp.end_time - temp.start_time
        min_task = None
        while temp:
            time = temp.end_time - temp.start_time
            if time <= min_time:
                min_time = time
                min_task = temp.task_id
            temp = temp.next
        return min_task, min_time
    
    #Function responsible for calculating average of the all execution times of the tasks in the linked list
    def get_average_time_of_all_tasks(self):
        temp = self.linked_list.head
        avg_time = 0
        sum_tasks = 0
        no_of_tasks = 0
        while temp:
            time = temp.end_time - temp.start_time
            sum_tasks += time
            no_of_tasks += 1
            temp = temp.next
        avg_time = sum_tasks / no_of_tasks
        return avg_time, sum_tasks, no_of_tasks