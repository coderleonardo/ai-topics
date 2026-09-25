## Client-Server Model

In this model, the client (an app, for example) sends a request to the server ("another computer" running some application), and the server responds with some message.

Note that the communication is bidirectional.

### Serving FastAPI locally

We can serve the FastAPI application locally and access this server from another computer using the IP of the application.

Finding the IP using Python:
```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
s.getsockname()[0]
```

## The Web Model

- Web
    - URL: internet address
    - HTTP: protocol of communication between devices
    - HTML: language used to create and structure web pages

### HTTP Message

#### Header

- Content-Type
- Authorization
- Accept
- Server

#### Body

Contains the data returned.

#### Verbs

When a client makes a request to the server, it needs to indicate what kind of action it wants to perform. These actions are conveyed by the *verbs*:

- GET: used to retrieve existing information from the server
- POST: used to create a new resource
- PUT: used to update an existing resource
- DELETE: used to delete a resource

#### Status Code

See IANA (Internet Assigned Numbers Authority).

- 1xx
- 2xx
- 3xx
- 4xx
- 5xx

Note that for validation, it's good practice to use constants to check the status code, like "OK = 200". See PEP-8 for more.

## API (Application Programming Interface)

Work following the client-server model where the client fetches information from the server via specific endpoints, respecting the HTTP rules. 

### Endpoint

It's a specific point where the client sends their requests. The location and structure of the endpoint define how the client formats its message. 

```python
@app.get("/")
def read_root():
    return {"message": "hello!"}
```

> The decorator `app.get('/')` indicates that when we use the method GET on the `/` endpoint the function `read_root` is called. 

### Documentation

It's good practice to write the API following the OpenAPI best practices. The documentation of the API can be visualized using the SwaggerUI for daily development (http://127.0.0.1:8000/docs) or ReDoc for final documentation (http://127.0.0.1:8000/redoc)

## JSON (JavaScript Object Notation)

```json 
{
    "livros": [
        {
            "titulo": "O apanhador no campo de centeio",
            "autor": "J.D. Salinger",
            "ano": 1945,
            "disponivel": false
        },
        {
            "titulo": "O mestre e a margarida",
            "autor": "Mikhail Bulgákov",
            "ano": 1966,
            "disponivel": true
        }
    ]
}
```

### Contracts in APIs JSON

When sharing information between the client and the server we need to make clear the structure of the data shared. To do this we use the `schema` property that specify:
- the response data
- additional restrictions
- structure of the objects

for example:
```json
{
  "type": "object",
  "properties": {
    "message": {
      "type": "string"
    }
  },
  "required": ["message"]
}
```

### Pydantic

Pydantic is a Python library that enables us to represent schemas through Python code to express JSON schemas and validate data types. 