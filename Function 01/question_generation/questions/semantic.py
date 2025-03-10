import random
from fractions import Fraction
from function_01.question_generation.resources.constants import (
    Lesson, 
    items_list,
    addition_structures,
    subtraction_structures,
    multiplication_structures,
    division_structures,
    DAYS,
    MONTHS,
    DAYS_IN_MONTH
)


# Number generating helper method
def question_numbers_generator(lesson):
    operations = {
        Lesson.ADDITION: (lambda: (random.randint(2, 10), random.randint(2, 10), lambda x, y: x + y)),
        Lesson.SUBTRACTION: (lambda: (random.randint(11, 20), random.randint(1, 10), lambda x, y: x - y)),
        Lesson.MULTIPLICATION: (lambda: (random.randint(2, 10), random.randint(2, 3), lambda x, y: x * y)),
        Lesson.DIVISION: (lambda: (lambda y: (y * (num2 := random.randint(2, 3)), num2, lambda x, y: x // y))(random.randint(2, 10)))
    }    
    num1, num2, operation = operations[lesson]()
    answer = operation(num1, num2)
    return num1, num2, answer


# 1. _________ Basic arithmetic questions _________
def arithmetic_question(lesson):    
    # Generate numbers and answer based on the lesson type
    num1, num2, answer = question_numbers_generator(lesson)    
    # Mapping of lesson types to operations and method strings
    methods = {
        Lesson.ADDITION: "plus",
        Lesson.SUBTRACTION: "minus",
        Lesson.MULTIPLICATION: "times",
        Lesson.DIVISION: "divided by"}
    
    method = methods[lesson]  # Get method name and operation
    question = f"What is {num1} {method} {num2}?"
    return question, answer


# 2. _________ Number between two numbers _________
def middle_number(lesson=None):
    answer = random.randint(1, 20)
    num1   = answer - 1
    num2   = answer + 1
    question = f"Which number comes between {num1} and {num2}?"
    return question, answer


# 3. _________ Number comparison _________
def number_question(lesson=None):
    num1 = random.randint(1, 20)
    num2 = random.randint(1, 20)
    if num1 == num2:
        num1 += 1
    
    comparison = "greater" if num1 > num2 else "smaller"
    question   = f"Which number is {comparison}, {num1} or {num2}?"
    return question, num1


# 4. _________ Context based arithmatic questions _________
def semantic_context_question(lesson):    
    # Generate numbers and answer based on the lesson type
    num1, num2, answer = question_numbers_generator(lesson)
    
    # Mapping of lesson types to structure strings list
    structure_lists = {
        Lesson.ADDITION: addition_structures,
        Lesson.SUBTRACTION: subtraction_structures,
        Lesson.MULTIPLICATION: multiplication_structures,
        Lesson.DIVISION: division_structures}
    
    structures = structure_lists[lesson]
    structure  = random.choice(structures)
    items = random.choice(items_list)

    question = structure.format(num1 = num1, num2 = num2, items = items)
    return question, answer


# 5. _________ Fraction question _________
def fraction_question(lesson=None):
    # Generate a random proper fraction (numerator < denominator)
    numerator = random.randint(1, 3)
    denominator = random.randint(numerator + 1, 4)  # Denominator greater than numerator
    fraction = Fraction(numerator, denominator)    
    # Generate a random number that is divisible by the denominator
    base_num = random.randint(2, 5)  # A base number to multiply by denominator
    num = base_num * denominator  # Ensuring num is divisible by denominator

    answer = int(fraction * num)    
    question = f"What is {fraction} of {num}?"    
    return question, answer


# 6. _________ Next odd / even number after num _________
def next_odd_even(lesson=None):
    num = random.randint(1, 100)
    parity = random.choice(["odd", "even"])
    
    if parity == "odd":
        answer = num + 1 if num % 2 == 0 else num + 2
    else:
        answer = num + 2 if num % 2 == 0 else num + 1
    
    question = f"What is the next {parity} number after {num}?"
    return question, answer


# 7. _________ Number of days in a month _________
def days_num_in_month(lesson=None):
    index = random.randint(0,11)
    month = MONTHS[index]
    question = f"How many days are there in {month}?"
    answer = DAYS_IN_MONTH[index]
    return question, str(answer)


# 8. _________ Day after number of days _________
def day_offset(lesson=None):
    index = random.randint(0,6)
    day = DAYS[index]
    direction = random.choice(["before", "after"])
    num = random.randint(2, 3)    
    
    if direction == "before":
        answer = DAYS[(index - num) % 7]
    else:
        answer = DAYS[(index + num) % 7]
    
    question = f"Which day is {num} days {direction} {day}?"
    return question, answer



# _________ Semantic and Verbal questions _________


# 9. _________ Day that comes before / after a day _________
def day_before_after(lesson=None):
    index = random.randint(0,6)
    day = DAYS[index]
    direction = random.choice(["before", "after"])
        
    if direction == "before":
        answer = DAYS[(index - 1) % 7]
    else:
        answer = DAYS[(index + 1) % 7]
    
    question = f"What is the day that comes {direction} {day}?"
    return question, answer


# 10. _________ Month that comes before / after a month _________
def month_before_after(lesson=None):
    index = random.randint(0,11)
    month = MONTHS[index]
    direction = random.choice(["before", "after"])   
    

    if direction == "before":
        answer = MONTHS[(index - 1) % 12]
    else:
        answer = MONTHS[(index + 1) % 12]
    
    question = f"What is the month that comes {direction} {month}?"
    return question, answer