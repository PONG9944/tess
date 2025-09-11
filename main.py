from fastapi import FastAPI

# Create the FastAPI application instance.
# The variable name must be `app` for Render to automatically detect it.
app = FastAPI()

@app.post("/endpoint")
def read_root(Va):
    return {"message": Va.data[0] * 10}
