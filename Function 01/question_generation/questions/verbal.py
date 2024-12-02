import random

# 1. What is the name of the number written as '[num]'?
def name_of_number(lesson=None):
    num = random.randint(1, 100)
    question = f"What is the name of the number written as '{num}'?"
    answer = str(num)
    return question, answer


# 2. What comes next in the sequence: [num1], [num2], [num3]?
def next_in_sequence(lesson=None):
    start = random.randint(1, 10)
    step = random.randint(1, 5)
    num1 = start
    num2 = start + step
    num3 = num2 + step
    next_num = num3 + step
    question = f"What comes next in the sequence: {num1}, {num2}, {num3}?"
    return question, str(next_num)


# 3. If someone says, [num1] [method] [num2], what operation is that?
def operation_name(lesson=None):
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    method = random.choice(("plus", "minus", "times", "divided by"))
    if method == "plus":
        operation = "addition"
    elif method == "minus":
        operation = "subtraction"
    elif method == "times":
        operation = "multiplication"
    elif method == "divided by":
        operation = "division"
    
    question = f"If someone says, {num1} {method} {num2}, what operation is that?"
    return question, operation


# 4. What number comes [after / before] '[num]'?
def before_after_number(lesson=None):
    num = random.randint(1, 100)
    direction = random.choice(["after", "before"])
    
    if direction == "after":
        answer = num + 1
    else:
        answer = num - 1
    
    question = f"What number comes {direction} '{num}'?"
    return question, str(answer)


# 5. Is [num1] greater than, less than, or equal to [num2]?
def compare_numbers(lesson=None):
    num1 = random.randint(2, 99)
    answer = random.choice(("greater than", "less than", "equal"))
    
    if answer == "less than":
        num2 = random.randint((num1+1),100)
    elif answer == "greater than":
        num2 = random.randint(1,(num1-1))
    else:
        num2 = num1
    
    question = f"Is {num1} greater than, less than, or equal to {num2}?"
    return question, answer


# 6. Is [num] an odd number or an even number?
def odd_or_even(lesson=None):
    num = random.randint(1, 100)
    answer = "odd" if num % 2 != 0 else "even"
    question = f"Is {num} an odd number or an even number?"
    return question, answer


# 7. How much time is there between [time] and [time]?
def time_difference(lesson=None):
    time1 = random.randint(1, 12)  # Hours between 0 and 23
    time2 = random.randint(1, 11)

    if time1 == time2:
        time2 = time2 + 1
        
    # Sort times to ensure time2 > time1
    time1, time2 = sorted([time1, time2])
    answer = f"{time2 - time1} hours"    
    question = f"How much time is there between {time1}:00 and {time2}:00?"
    return question, answer