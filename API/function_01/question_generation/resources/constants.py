from enum import Enum

# Question Types
class DyscalculiaType(Enum):
    PROCEDURAL = "procedural"
    SEMANTIC   = "semantic"
    VERBAL     = "verbal"

# Difficulty levels using Enum
class Difficulty(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"
    
# Lessons using Enum
class Lesson(Enum):
    ADDITION       = "addition"
    SUBTRACTION    = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION       = "division"
    NUMBERS        = "dumbers"
    COMPARISON     = "comparison"
    ODDEVEN        = "odd_even"
    DAYS           = "days"
    MONTHS         = "months"
    FRACTION       = "fraction"
    TIME           = "time"
    MATHWORDS      = "math_words"
    

SHAPES    = ("triangle", "square", "rectangle", "circle")
DAYS      = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS    = ("January", "February", "March", "April","May", "June", "July", "August",
             "September", "October", "November", "December"
            )
DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)



# Combinations where the sum is >= 10
sum_greater_or_equal_ten = (
    (1, 9), (2, 8), (2, 9), (3, 7), (3, 8), (3, 9), (4, 6), (4, 7), (4, 8), 
    (4, 9), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (6, 4), (6, 5), (6, 6), 
    (6, 7), (6, 8), (6, 9), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), 
    (7, 9), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), 
    (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)
)

# Combinations where the sum is < 10
sum_less_than_ten = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), 
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 1), (3, 2), (3, 3), 
    (3, 4), (3, 5), (3, 6), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 1), 
    (5, 2), (5, 3), (6, 1), (6, 2), (7, 1), (8, 1),
)

# Item names to create questions 
items_list = (
    "apples", "oranges", "bananas", "books", "pencils", "pens", "chocolates", 
    "cookies", "stickers", "candies", "toy cars", "toys", "balls", "cups", 
    "mugs", "markers", "papers", "balloons"
)

addition_structures = (
    "If you have {num1} {items} and you add {num2} more, how many {items} do you have?",
    "Tom has {num1} {items}. He gets {num2} more as a gift. How many {items} does he have now?",
    "Alex has {num1} {items}. He buys {num2} more {items}. How many {items} does he have now?",
    "Sarah has {num1} {items}. She borrows {num2} more {items} from a friend. How many books does she have now?"
)

subtraction_structures = (
    "If there are {num1} {items} and you take away {num2}, how many {items} do you have left?",
    "You have {num1} {items} and you give {num2} to your friend. How many {items} do you have left?",
)

multiplication_structures = (
    "If there are {num1} bags with {num2} {items} each, how many {items} are there in total?",
)

division_structures = (
    "If you have {num1} {items} and want to share them among {num2} friends, how many does each friend get?",
)

combined_structures = (
    "If you bought {num1} {items1} for {num2} each and then spent an additional {num3} on {items2}, what is the total amount spent?",
)