# Generative AI Applications with Amazon Bedrock

## Module 01 - Knowlodge Bases and Workflows

### RAG Overview

![RAG Workflow](./images/rag-workflow.png)

Amazon Bedrock Knowledge Bases enhance foundation models with private data using **Retrieval-Augmented Generation (RAG)**, allowing you to get contextually relevant responses without fine-tuning a model. This is especially useful for making an organization's proprietary information, like internal runbooks or customer support documents, accessible and queryable.

-----

### **How Knowledge Bases Works: The RAG Approach** 🧠

Knowledge Bases uses RAG to connect a foundation model to your private data. This process involves three main steps:

1.  **Document Processing:** Your documents are uploaded to a data source (like Amazon S3), chunked into smaller pieces, and then converted into **embeddings**—numerical representations of text. This is done with an embedding model like Amazon Titan Text Embeddings.
2.  **Vector Storage:** The embeddings are stored in a **vector database** (e.g., Amazon OpenSearch Serverless). This allows the system to perform **semantic search**, finding content based on meaning rather than just keywords.
3.  **Query Processing:** When a user asks a question, it's converted into an embedding. The system then searches the vector database for similar embeddings and retrieves the most relevant content. Finally, a foundation model uses this retrieved content to generate a response, which includes **citations** to the source documents.

-----

### **Setting Up Your Knowledge Base** ⚙️

Setting up a knowledge base involves several configuration steps:

  * **Initial Setup:** You provide a name for the knowledge base, and Bedrock automatically handles the creation of the necessary **IAM roles** to manage permissions between services.
  * **Configure Data Source:** You specify where your documents are stored. While Amazon S3 is a common choice, Bedrock supports other sources like **Confluence**, **Salesforce**, and **SharePoint**.
  * **Choose Parsing Strategy:** This determines how content is extracted from your documents. The **default parser** works for common files like PDFs, while **Bedrock Data Automation** can handle more complex, multimodal content.
  * **Select Chunking Strategy:** Chunking breaks documents into manageable pieces for retrieval. The default strategy uses 300-token chunks, but you can choose other methods based on your document's structure.
    - Default: Chunks content into manageable pieces of around 300 tokens while respecting sentence boundaries.

    - Fixed Size: Divides documents into equal-sized chunks, useful for consistent processing.

    - Hierarchical: Chunks are created at multiple levels (e.g., paragraphs, sections, or entire documents), which helps with queries that require broad or specific context.

    - Semantic: Chunks are created based on semantic similarity, ensuring that related information stays together.

    - No Chunking: The entire document is treated as a single chunk, which is only suitable for very short documents.
  * **Configure Models and Storage:** You need to select an **embedding model** (e.g., Amazon Titan Text Embeddings) and a **vector database** (e.g., Amazon OpenSearch Serverless) to store the embeddings.

After setup, you **sync your data source** to generate the embeddings. Once complete, you can use the `RetrieveAndGenerate` API to query the knowledge base and receive responses based on your private data.

```python
response = bedrock.retrieve_and_generate(        
     input={"text": "How do I restart the staging environment?"},        
     retrieveAndGenerateConfiguration={            
          "type": "KNOWLEDGE_BASE",            
          "knowledgeBaseConfiguration": {                
               "knowledgeBaseId": "<KNOWLEDGE_BASE_ID>",                 
               "modelArn": "<MODEL_ID>"            
             }
        }
    )
```

-----

### **Real-World Applications and Summary** 🎯

Knowledge Bases can be applied to many use cases, such as:

  * DevOps documentation search
  * Customer support systems
  * Sales enablement
  * HR assistance

For a customer support system, a knowledge base can instantly search through thousands of documents to provide accurate answers to customer questions, reducing the need for manual searches and escalations. The effectiveness of a knowledge base depends on how well you structure your documents and configure your settings. Experimenting with different chunking and parsing strategies is key to getting the best results.

### Prompt Management

Amazon Bedrock **Prompt Management** is a feature that centralizes and simplifies the creation, versioning, and reuse of prompts for foundation models. It helps developers avoid embedding prompts directly in their application code, which makes managing and updating prompts much easier.

-----

#### **What is Prompt Management?**

Prompt Management allows you to:

  * **Create and save** prompts as reusable templates.
  * Use **version control** to manage changes to your prompts over time.
  * Include **variables** (placeholders) for dynamic content.
  * Manage prompts across different environments (development, staging, production).

-----

#### **Key Components**

##### **Prompt Structure**

A prompt in Prompt Management is a named resource that can contain one or more **variants**. It is defined by its name, a description, a default variant, and a list of variants. The core of the prompt is the template text, which can include variables enclosed in double curly braces (e.g., `{{job_title}}`).

  * **Example Code:**
    ```python
    import boto3
    client = boto3.client('bedrock-agent')
    create_response = client.create_prompt(
        name='hr_assistant_prompt_template',
        description="Generates inclusive job descriptions from structured inputs",
        defaultVariant="v1",
        variants=[
            {
                "name": "v1",
                "modelId": "<MODEL_ID>",
                "templateType": "TEXT",
                "templateConfiguration": {
                    "text": {
                        "inputVariables": [
                            {"name": "job_title"},
                            {"name": "responsibilities"},
                            {"name": "requirements"},
                            {"name": "location"},
                            {"name": "work_type"}
                        ],
                        "text": '''
    You are an HR assistant.
    Write a professional, inclusive job description using the following inputs:
    Job title: {{job_title}}
    Responsibilities: {{responsibilities}}
    Requirements: {{requirements}}
    Location: {{location}}
    Work type: {{work_type}}
    - Start with a clear summary
    - Use concise, inclusive language
    - Keep it under 250 words
    '''
                    }
                },
                "inferenceConfiguration": {
                    "text": {
                        "maxTokens": 500,
                        "temperature": 0.7,
                        "topP": 0.9,
                        "stopSequences": []
                    }
                }
            }
        ]
    )
    ```

##### **Variants**

A prompt can have multiple variants, each with its own configuration. This is useful for A/B testing or for providing different behaviors for the same task. Each variant includes:

  * A **variant name**.
  * A **model ID**.
  * **Inference configuration** (e.g., `temperature`, `topP`).
  * **Template configuration**.
  * **Example Code:**
    ```python
    variant_examples = [
        # First variant for a detailed response
        {
            "name": "detailed",
            "modelId": "<MODEL_ID>",
            "inferenceConfiguration": {
                "text" : {
                  "temperature": 0.2,  # Lower temperature for a more deterministic response
                  "topP": 0.9
                }
            }
        },
        # Second variant for a summary response
        {
            "name": "summary",
            "modelId": "<MODEL_ID>",
            "inferenceConfiguration": {
                "text" : {
                 "temperature": 0.7,  # Higher temperature for more creative responses
                 "topP": 0.9
                }
            }
        }
    ]
    ```

##### **Template Types**

Prompt Management supports two types of templates:

  * **`TEXT` Template:** Designed for single text message prompts.
  * **`CHAT` Template:** Designed for conversational formats and compatible with models that support the `Converse` API. This template type is required for using prompt caching.

-----

#### **Using Prompt Templates**

Once a prompt is created, you can use its ARN to invoke it via the Bedrock API, passing in values for any defined variables.

  * **Example Code (using `Converse` API):**
    ```python
    response = bedrock.converse(
         modelId="<prompt_arn>",
         promptVariables={
              'request': {
                   'text': "<request_content_here>"
            }
        },
    )
    ```

#### Example

##### **1. Create the Prompt**

First, you define your prompt template using the `create_prompt` API. This involves setting up the prompt's name, a description, and at least one **variant**. This example uses a text-based prompt for a job description generator. It includes input variables like `job_title` and `responsibilities` that will be replaced with dynamic content later.

```python
import boto3
import json

bedrock_agent_client = boto3.client('bedrock-agent')

prompt_name = "job_description_generator"
template_text = """
You are an HR assistant.
Write a professional, inclusive job description using the following inputs:
Job title: {{job_title}}
Responsibilities: {{responsibilities}}
Requirements: {{requirements}}
Location: {{location}}
Work type: {{work_type}}

- Start with a clear summary.
- Use concise, inclusive language.
- Keep it under 250 words.
"""

response = bedrock_agent_client.create_prompt(
    name=prompt_name,
    description="Generates inclusive job descriptions from structured inputs",
    defaultVariant="v1",
    variants=[
        {
            "name": "v1",
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "templateType": "TEXT",
            "templateConfiguration": {
                "text": {
                    "inputVariables": [
                        {"name": "job_title"},
                        {"name": "responsibilities"},
                        {"name": "requirements"},
                        {"name": "location"},
                        {"name": "work_type"}
                    ],
                    "text": template_text
                }
            },
            "inferenceConfiguration": {
                "text": {
                    "maxTokens": 500,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
        }
    ]
)
print(f"Prompt '{prompt_name}' created with ARN: {response['prompt']['arn']}")
```

-----

##### **2. Update the Prompt (Versioning)**

After creating the initial prompt, you might want to refine it. For example, you may want to create a more concise version for a mobile app. Instead of editing the original, you can add a new **variant** using `update_prompt`. This allows you to A/B test different versions or use them for different purposes while keeping them under a single logical prompt.

```python
# Assuming you have the prompt ARN from the creation step
prompt_arn = response['prompt']['arn']

# Add a new variant for a more concise version
bedrock_agent_client.update_prompt(
    promptIdentifier=prompt_arn,
    defaultVariant="v1",
    variants=[
        {
            "name": "v1", # Keep the original variant
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "templateType": "TEXT",
            "templateConfiguration": {
                "text": {
                    "inputVariables": [
                        {"name": "job_title"},
                        {"name": "responsibilities"},
                        {"name": "requirements"},
                        {"name": "location"},
                        {"name": "work_type"}
                    ],
                    "text": template_text
                }
            },
            "inferenceConfiguration": {
                "text": {
                    "maxTokens": 500,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
        },
        {
            "name": "concise", # Add a new, more concise variant
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "templateType": "TEXT",
            "templateConfiguration": {
                "text": {
                    "inputVariables": [
                        {"name": "job_title"},
                        {"name": "responsibilities"},
                        {"name": "requirements"}
                    ],
                    "text": """
                    Generate a short, professional job description for {{job_title}}. 
                    Highlight key responsibilities: {{responsibilities}} 
                    and requirements: {{requirements}}.
                    """
                }
            },
            "inferenceConfiguration": {
                "text": {
                    "maxTokens": 200,
                    "temperature": 0.5,
                    "topP": 0.9
                }
            }
        }
    ]
)
print(f"Prompt '{prompt_name}' updated with new 'concise' variant.")
```

-----

##### **3. Use the Prompt in Your Application**

Instead of hard-coding the prompt text, you can now call the `invoke_prompt` API using the prompt's ARN and specify which variant you want to use. This makes your application flexible, allowing you to switch between variants without changing a single line of application code.

```python
import boto3
import json

bedrock_agent_client = boto3.client('bedrock-agent')

# Define the input variables
input_variables = {
    'job_title': 'Senior Software Engineer',
    'responsibilities': 'Design, develop, and maintain backend services; collaborate with product managers; mentor junior engineers.',
    'requirements': '5+ years of experience; proficiency in Python and AWS; strong problem-solving skills.',
    'location': 'Remote',
    'work_type': 'Full-time'
}

# Invoke the prompt using the `v1` variant
response_v1 = bedrock_agent_client.invoke_prompt(
    promptIdentifier=prompt_arn,
    promptVariant="v1",
    promptVariables=input_variables
)

# Invoke the prompt using the `concise` variant
response_concise = bedrock_agent_client.invoke_prompt(
    promptIdentifier=prompt_arn,
    promptVariant="concise",
    promptVariables=input_variables
)

print("--- Generated Job Description (V1) ---")
print(response_v1['text'])

print("\n--- Generated Job Description (Concise) ---")
print(response_concise['text'])
```

### **Amazon Bedrock Converse APIs**

The Amazon Bedrock **Converse API** is designed for building sophisticated conversational applications by maintaining **context** across multiple interactions. Unlike stateless API calls, it provides a unified and structured way to manage conversation history, making it ideal for chatbots and virtual assistants. It also simplifies model switching by offering a consistent interface across different FMs.

-----

#### **Building Blocks of the Converse API**

The Converse API is composed of several key components that work together to manage a conversation.

  * **System Prompt:** A system prompt defines the model's personality, behavior, and overall operating parameters. It sets the foundational rules for the entire conversation.

      * **Example:** `You are an AI assistant that helps create playlists. Only return song names and artists.`

  * **Message Array:** This component stores the entire conversation history as an ordered sequence of messages. Each message includes a `role` (`user` or `assistant`) and the message `content`.

      * **Example:**
        ```json
        [
            {
                "role": "user",
                "content": "Create a playlist of 3 pop songs"
            }
        ]
        ```

  * **Inference Parameters:** These parameters control how the model generates its response, influencing its creativity and output length. Supported parameters can vary by model.

      * **Example for Claude:**
        ```json
        {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.999
        }
        ```

  * **Tool Configuration:** This allows the model to interact with external functions to perform real-world actions. The tool is defined with its name, a description, and an `inputSchema`.

      * **Example:**
        ```json
        {
            "toolSpec": {
                "name": "getWeather",
                "description": "Gets the current weather using latitude and longitude.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "lat": {"type": "number"},
                            "lon": {"type": "number"}
                        },
                        "required": ["lat", "lon"]
                    }
                }
            }
        }
        ```

-----

#### **Important Implementation Considerations**

  * **Context Window Management:** The **context window** is a critical limitation that defines how much information the model can process at once. For long conversations, you need to manage this by monitoring token usage and implementing strategies like **conversation summarization** or **pruning** to avoid exceeding the model's limit.
  * **Production Best Practices:** For production applications, conversation histories should be stored in a persistent database (like DynamoDB or Redis) rather than in memory. Proper **session management** and monitoring are essential for handling user interactions, rate limits, and logging.
  * **Model-Specific Considerations:** Different FMs have unique capabilities and requirements. Always check the model-specific documentation, as some models may not support system prompts or certain inference parameters.

-----

#### **Tool Execution Flow**

When a model needs to use a tool, it initiates a specific multi-step flow:

1.  Your application sends the user's message and the available tool definitions to the model.
2.  The model responds with a **`toolUse` block** if it decides to use a tool.
3.  Your application detects this `toolUse` block and extracts the tool parameters.
4.  Your application **executes the tool** (e.g., calls a Lambda function).
5.  Your application sends the **tool's result** back to the model as a new user message.
6.  The model receives the tool output and generates a final, comprehensive response for the user.

<!-- end list -->

  * **Example Code for Tool Execution Flow:**
    ```python
    import boto3
    import json

    bedrock = boto3.client(service_name='bedrock-runtime')
    lambda_client = boto3.client("lambda")
    MODEL_ID_CLAUDE_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"

    # Define toolSpec for lambda function
    weather_tool = {
        "toolSpec": {
            "name": "getWeather",
            "description": "Gets the current weather using latitude and longitude.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"}
                    },
                    "required": ["lat", "lon"]
                }
            }
        }
    }

    # Invoke Lambda Function (tool)
    def invoke_weather_lambda(input_data):
        response = lambda_client.invoke(
            FunctionName="weather-tool",
            InvocationType="RequestResponse",
            Payload=json.dumps(input_data)
        )
        payload = response["Payload"].read()
        result = json.loads(payload)
        body = result.get("body", "{}")
        return json.loads(body) if isinstance(body, str) else body

    initial_user_msg = {
        "role": "user",
        "content": [{"text": "What's the weather like in portland?"}]
    }

    system_prompt = [
        {"text": """
        You are a helpful assistant that can check the weather using tools when needed.
         If the user inputs a city name that exists in multiple geogrpahic areas, ask the user for clarification.
        """}]

    # Ask model initial question
    initial_response = bedrock.converse(
        modelId=MODEL_ID_CLAUDE_SONNET,
        system=system_prompt,
        messages=[initial_user_msg],
        toolConfig={
            "tools": [weather_tool],
            "toolChoice": {"auto": {}}
        },
        inferenceConfig={"temperature": 0.7}
    )

    # Parse response from Claude to determine if tool is needed
    assistant_msg = initial_response["output"]["message"]
    content_blocks = assistant_msg["content"]
    tool_use_block = next((block for block in content_blocks if "toolUse" in block), None)

    if not tool_use_block:
        print("=== Claude Response ===")
        print(content_blocks[0]["text"])
    else:
        tool_use = tool_use_block["toolUse"]
        tool_input = tool_use["input"]
        tool_use_id = tool_use["toolUseId"]
        print(tool_use_block)
        print(f"→ Claude requested tool: getWeather with input: {tool_input}")

        # Invoke tool
        tool_output = invoke_weather_lambda(tool_input)
        print(f"← Lambda returned: {tool_output}")

        # Print out tool output
        try:
            summary = f"The weather in {tool_output['location']} is currently {tool_output['temperature']} with {tool_output['condition']}."
        except Exception as e:
            summary = f"Sorry, I couldn't retrieve the weather. ({str(e)})"

        # Construct tool result message
        tool_result_msg = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": summary}]
                    }
                }
            ]
        }

        # Pass tool result message to model
        final_response = bedrock.converse(
            modelId=MODEL_ID_CLAUDE_SONNET,
            messages=[initial_user_msg, assistant_msg, tool_result_msg],
            toolConfig={
                  "tools": [weather_tool],
                "toolChoice": {"auto": {}}
            },
            inferenceConfig={"temperature": 0.7}
        )

        # Print final response
        final_msg = final_response["output"]["message"]["content"][0]["text"]
        print("\n=== Final Agent Response ===")
        print(final_msg)
    ```

### Amazon Bedrock Flows

Amazon Bedrock Flows is a visual workflow builder that lets developers create generative AI applications without writing a lot of code. It uses a drag-and-drop interface to connect different components, allowing you to build and manage complex AI-powered workflows.

***

#### **Flow Control**

Flow control components manage the flow of data and logic within a Bedrock Flow.

* **Input and Output Nodes:** These are the entry and exit points of your workflow. The **Flow Input Node** defines what data the flow accepts, while the **Flow Output Node** extracts and formats the final data using a query language.
* **Flow Logic Controllers:** These nodes allow for sophisticated logic.
    * **Condition Node:** Implements branching logic, evaluating conditions to send data down different paths.
    * **Iterator Node:** Processes items within an array one by one, enabling sequential or parallel operations on collections of data.
    * **Collector Node:** Aggregates processed items back into a single array or combines results from different branches.

***

#### **Data Processing**

Data processing components handle the core computational work and AI interactions within the flow.

* **AI Integration:** The **Prompt Node** is the main way to interact with Bedrock's foundation models. You can use this node to create prompts, use saved templates from your prompt library, and configure inference parameters to get the desired output. 
* **Service Integration:** Bedrock Flows can connect to other AWS services through specialized nodes, allowing you to integrate custom logic and data sources:
    * **Amazon S3** for storing and retrieving data.
    * **AWS Lambda** for running custom code and business logic.
    * **Amazon Lex** for building conversational interfaces.
    * **Bedrock Agents and Knowledge Bases** for enhancing the AI capabilities with tools and private data.

***

### **Summary**

Amazon Bedrock Flows simplifies generative AI development by providing a visual, no-code interface. You can create complex workflows by dragging and dropping nodes that represent Bedrock features and other AWS services. Every flow follows a pattern of **input**, **processing**, and **output**, giving you the building blocks needed to create powerful, scalable AI applications.