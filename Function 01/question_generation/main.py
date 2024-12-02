from resources.constants import DyscalculiaType, Difficulty, Lesson
from question_generator.get_question import get_random_question

import uvicorn 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Define the request model using Pydantic
class QuestionRequest(BaseModel):
    dyscalculia_type: str
    lesson: str = None # Optional lesson
    difficulty: str

@app.post("/generate-question")
def generate_question(request: QuestionRequest):
    try:
        # Convert request data (strings) to corresponding enum types
        dyscalculia_type = DyscalculiaType[request.dyscalculia_type]
        lesson = Lesson[request.lesson] if request.lesson else None
        difficulty = Difficulty[request.difficulty]
                
        generated_question = get_random_question(dyscalculia_type, difficulty, lesson)
        
        # Return the generated question in JSON format
        return {"question": generated_question}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid key: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn main:app --reload
