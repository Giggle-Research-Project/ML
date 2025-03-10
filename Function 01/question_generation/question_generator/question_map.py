from function_01.question_generation.questions import procedural, semantic, verbal
from function_01.question_generation.resources.constants import DyscalculiaType, Difficulty, Lesson


question_map = {
    DyscalculiaType.PROCEDURAL: {
        Lesson.ADDITION: procedural.addition_question,
        Lesson.SUBTRACTION: procedural.subtraction_question,
        Lesson.MULTIPLICATION: procedural.multiplication_question,
        Lesson.DIVISION: procedural.division_question,
    },
    DyscalculiaType.SEMANTIC: {
        Difficulty.EASY: {
            Lesson.ADDITION: semantic.arithmetic_question,
            Lesson.SUBTRACTION: semantic.arithmetic_question,
            Lesson.MULTIPLICATION: semantic.arithmetic_question,
            Lesson.DIVISION: semantic.arithmetic_question,
            Lesson.NUMBERS: semantic.middle_number,
            Lesson.COMPARISON: semantic.number_question,                    
        },
        Difficulty.MEDIUM: {
            Lesson.ADDITION: semantic.semantic_context_question, # Addition
            Lesson.SUBTRACTION: semantic.semantic_context_question, # Subtraction
            Lesson.MULTIPLICATION: semantic.semantic_context_question, # Multiplication
            Lesson.DIVISION: semantic.semantic_context_question, # Division
            Lesson.ODDEVEN: semantic.next_odd_even,
            Lesson.DAYS: semantic.day_before_after,
        },                
        Difficulty.HARD: {
            Lesson.FRACTION: semantic.fraction_question, 
            Lesson.MONTHS: semantic.month_before_after, 
            Lesson.DAYS: (semantic.day_offset, semantic.days_num_in_month),
        },               
    },
    DyscalculiaType.VERBAL: {
        Difficulty.EASY: {
            Lesson.NUMBERS: (verbal.name_of_number, verbal.before_after_number)                
        },
        Difficulty.MEDIUM: {
            Lesson.NUMBERS: verbal.next_in_sequence, 
            Lesson.COMPARISON: verbal.compare_numbers,
            Lesson.ODDEVEN: verbal.odd_or_even,
            Lesson.TIME: verbal.time_difference,
        },                
        Difficulty.HARD: {
            Lesson.MATHWORDS: (verbal.operation_name),
        },        
    },
}

    