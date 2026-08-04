from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from http import HTTPStatus


app = FastAPI()


@app.get(
    "/", 
    status_code=HTTPStatus.OK, 
    response_class=HTMLResponse
)  # expose the function to be served by FastAPI
def read_root():
    # return {"message": "hello everyone"}
    return """
    <html>
        <head>
            <title> hello everyone </title>
        </head>
        <body>
            <h1> hello everybody!!! </h1>
        </body>
    </html>"""