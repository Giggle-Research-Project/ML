from function_01.question_generation.resources.constants import DyscalculiaType, Difficulty, Lesson
from function_01.question_generation.question_generator.question_map import question_map

import random

def get_random_question(dyscalculia_type: DyscalculiaType, difficulty: Difficulty, lesson: Lesson):
    try:
        # Navigate to the correct section of the question_types
        if dyscalculia_type == DyscalculiaType.PROCEDURAL:
            # For procedural, directly access lesson
            question_function = question_map[DyscalculiaType.PROCEDURAL][lesson]
            return question_function(difficulty)
        
        else:
            # For semantic and verbal, access by difficulty first
            question_dict = question_map[dyscalculia_type][difficulty]

            # Check if the lesson exists in the difficulty level
            if lesson in question_dict:
                question_function = question_dict[lesson]
                
                # If multiple functions are available, randomly select one
                if isinstance(question_function, tuple):
                    # Randomly select a function if it's a tuple
                    question_function = random.choice(question_function)
                
                # Call the question function to get a question
                return question_function(lesson)
            else:
                raise ValueError(f"Lesson \"{lesson}\" not found for difficulty \"{difficulty}\" in \"{dyscalculia_type}\".")

    except KeyError as e:
        raise ValueError(f"Invalid type or difficulty: {e}")