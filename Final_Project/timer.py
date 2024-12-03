"""
timer.py

This module provides utility functions to measure the execution time of a program.
It includes functions to start a timer and to calculate and print the elapsed time
in a human-readable format.

Functions:
- start_timer(): Starts a timer and returns the start time.
- get_time_passed(start_time): Calculates and prints the elapsed time since the start time.
"""

import time

def start_timer():
    """
    Starts a timer and returns the current time.

    Returns:
    - float: The current time in seconds since the Epoch.
    """
    return time.time()

def get_time_passed(start_time):
    """
    Calculates and prints the elapsed time since the start time.

    Parameters:
    - start_time (float): The start time in seconds since the Epoch.

    Prints:
    - str: The elapsed time in minutes and seconds if minutes > 0, otherwise in seconds.
    """
    end_time = time.time()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    if minutes > 0:
        print(f"Program completed in {int(minutes)} minutes and {seconds:.2f} seconds\n")
    else:
        print(f"Program completed in {seconds:.2f} seconds\n")
        