import datetime

def str_date():
    # returns the date as a string formatted as <year>-<month>-<day>
    a = str(datetime.datetime.now())
    b = a.split(" ")[0]
    return b