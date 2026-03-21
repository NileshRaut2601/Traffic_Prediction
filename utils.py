import pandas as pd

def classify_traffic_level(vehicles):
    """
    Classify traffic level based on vehicle count.
    
    Args:
        vehicles (int): Number of vehicles
        
    Returns:
        str: Traffic level ('Low', 'Medium', 'High')
    """
    if vehicles < 20:
        return 'Low'
    elif 20 <= vehicles <= 50:
        return 'Medium'
    else:
        return 'High'

def get_days_in_month(year, month):
    """
    Return number of days in a given month and year, including leap-year logic.
    """
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month in [4, 6, 9, 11]:
        return 30
    # February
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 29
    return 28


def preprocess_input(junction, year, month, day, weekday, hour):
    """
    Preprocess user input into a DataFrame for prediction.
    
    Args:
        junction (int): Junction number
        year (int): Year
        month (int): Month (1-12)
        day (int): Day (1-31)
        weekday (int): Weekday (0=Monday, 6=Sunday)
        hour (int): Hour (0-23)
        
    Returns:
        pd.DataFrame: Preprocessed input data
    """
    data = {
        'Junction': [junction],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Weekday': [weekday],
        'Hour': [hour]
    }
    return pd.DataFrame(data)