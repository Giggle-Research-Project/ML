import random
from function_01.question_generation.resources.constants import Difficulty, sum_greater_or_equal_ten, sum_less_than_ten

# 1. _________ Addition question _________

def addition_question(difficulty):
    if difficulty == Difficulty.EASY:
        ones = random.choice(sum_less_than_ten)
        tens = random.choice(sum_less_than_ten)
        
    elif difficulty == Difficulty.MEDIUM:
        ones = random.choice(sum_greater_or_equal_ten)
        tens = random.choice(sum_less_than_ten)
        
    else:
        ones = random.choice(sum_greater_or_equal_ten)
        tens = random.choice(sum_greater_or_equal_ten)
        hundreds = random.choice(sum_less_than_ten)

    # Construct numbers
    num1 = ''.join(map(str, [hundreds[0] if difficulty == Difficulty.HARD else '', tens[0], ones[0]]))
    num2 = ''.join(map(str, [hundreds[1] if difficulty == Difficulty.HARD else '', tens[1], ones[1]]))
    
    answer = int(num1) + int(num2)    
    return (num1, num2, answer)
    
    
# 2. _________ Subtraction question _________  

# Generate two unequal numbers for subtraction questions
def generate_two_unequals():
    num1, num2 = random.randint(2, 9), random.randint(2, 9)
    if num1 == num2: num1 -= 1
    return (max(num1, num2), min(num1, num2))

# Subtraction question generator   
def subtraction_question(difficulty):
    ones = generate_two_unequals()
    tens = generate_two_unequals()
        
    if difficulty == Difficulty.EASY:      
        minuend    = str(tens[0]) + str(ones[0])  
        subtrahend = str(tens[1]) + str(ones[1]) 
        
    elif difficulty == Difficulty.MEDIUM:      
        minuend    = str(tens[0]) + str(ones[1])  
        subtrahend = str(tens[1]) + str(ones[0])  
        
    else:
        hundreds   = generate_two_unequals()   
        minuend    = str(hundreds[0]) + str(tens[1]) + str(ones[1])
        subtrahend = str(hundreds[1]) + str(tens[0]) + str(ones[0]) 
    
    answer = int(minuend) - int(subtrahend)    
    return minuend, subtrahend, answer



# 3. _________ Multiplication question _________

def multiplication_question(difficulty):
    if difficulty == Difficulty.EASY:
        multipliers = (2,3)        
    elif difficulty == Difficulty.MEDIUM:
        multipliers = (4,5)        
    else:
        multipliers = (6,7)

    multiplicand = random.randint(10, 99)
    multiplier   = random.choice(multipliers)
    
    answer = multiplicand * multiplier    
    return multiplicand, multiplier, answer



# 4. _________ Devision question _________

# long Division Answers
def long_division(dividend, divisor):
    dividend_1 = dividend // 10  # Get the first digit (tens place)
    dividend_2 = dividend % 10   # Get the second digit (ones place)

    b1 = dividend_1 // divisor
    b2 = b1 * divisor
    b3 = dividend_1 % divisor
    b4 = dividend_2
    new_dividend = b3 * 10 + b4
    b5 = new_dividend // divisor
    b8 = new_dividend % divisor
    temp = b5 * divisor
    b6 = temp // 10  # Get the first digit (tens place)
    b7 = temp % 10   # Get the second digit (ones place)        

    # if dividend_1 > divisor:
    #     answers = (b1,b2,b3,b4,b5,b6,b7,b8)
    # else:
    #     answers = (b5,b6,b7,b8)

    # if we use only one long structure
    answers = (b1,b2,b3,b4,b5,b6,b7,b8)   
    return answers

# Devision question generator
def division_question(difficulty):
    if difficulty == Difficulty.EASY:
        divisors = (2,3)        
    elif difficulty == Difficulty.MEDIUM:
        divisors = (4,5)        
    else:
        divisors = (6,7)

    dividend = random.randint(10, 99)
    divisor  = random.choice(divisors)
    quotient = dividend // divisor
    remainder = dividend % divisor
    long_division_answers  = long_division(dividend, divisor)    
    return dividend, divisor, quotient, remainder, long_division_answers