from fastapi import FastAPI
from pydantic import BaseModel, conlist
import uvicorn

# Create the FastAPI application instance.
# The variable name must be `app` for Render to automatically detect it.
app = FastAPI()

@app.post("/endpoint")
def read_root(Va):
    return {"message": Va.data[0] * 10}





# 1. Data Model: This ensures the incoming data is a list of 13 integers.
#    `conlist` is a Pydantic feature that adds a size constraint.
class FoodData(BaseModel):
    tags: conlist(int, min_length=1, max_length=1)


@app.post("/endpoint")
async def predict_favorability(food_data: FoodData):
    return {"RES": 3}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
