import random

def get_color():
    x = random.randint(0, 100)

    if x < 25:
        return 'green'
    elif x < 50:
        return 'yellow'
    elif x < 75:
        return 'orange'
    else:
        return 'red'
