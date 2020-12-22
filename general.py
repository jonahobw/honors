import datetime

def str_date():
    # returns the date as a string formatted as <year>-<month>-<day>
    a = str(datetime.datetime.now())
    b = a.split(" ")[0]
    return b

def format_two_digits(number):
    # parameters:
    # number (int, float, or string): number to be converted
    #
    # return values:
    # two_digits (string): the input number converted to a string and padded
    # with zeros on the front so that it is at least 2 digits long
    return str(number).zfill(2)