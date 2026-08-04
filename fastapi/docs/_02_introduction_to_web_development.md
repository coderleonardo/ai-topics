## Client-Server Model

In this model, the client (an app, for example) sends a request to the server ("another computer" running some application), and the server responds with some message.

Note that the communication is bidirectional.

### Serving FastAPI locally

We can serve the FastAPI application locally and access this server from another computer using the IP of the application.

Finding the IP using Python:
```python
import socket 

s = socket.socket(socket.AF_INET, socket.SOCK.DGRAM)
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