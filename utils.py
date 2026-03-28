import pandas as pd

def classify_traffic_level(vehicles):
    if vehicles < 20:
        return 'Low'
    elif vehicles <= 50:
        return 'Medium'
    else:
        return 'High'


def get_days_in_month(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month in [4, 6, 9, 11]:
        return 30
    return 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28


def convert_to_24_hour(hour, period):
    return 0 if (hour == 12 and period == "AM") else \
           12 if (hour == 12 and period == "PM") else \
           hour if period == "AM" else hour + 12


def preprocess_input(junction, year, month, day, weekday, hour):
    return pd.DataFrame([{
        'Junction': junction,
        'Year': year,
        'Month': month,
        'Day': day,
        'Weekday': weekday,
        'Hour': hour
    }])