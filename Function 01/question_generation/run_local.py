from question_generator.get_question import get_random_question
from resources.constants import DyscalculiaType, Difficulty, Lesson

def main():
    try:
        # Convert request data (strings) to corresponding enum types
        dyscalculia_type = DyscalculiaType.VERBAL
        lesson = Lesson.MATHWORDS
        difficulty = Difficulty.HARD
                
        generated_question = get_random_question(dyscalculia_type, difficulty, lesson)
        
        # Return the generated question in JSON format
        return {"question": generated_question}
    
    except ValueError as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    print(main())