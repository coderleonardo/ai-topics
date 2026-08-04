from fastapi import FastAPI

app = FastAPI()


@app.get("/")  # expose the function to be served by FastAPI
def read_root():
    return {"message": "hello everyone"}
