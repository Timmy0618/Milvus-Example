# Milvus Example

This is a Milvus combine LLM(ollama) practice.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- Docker
- Docker Compose
- Python 3.6 or higher

### Installing

A step-by-step series of examples that tell you how to get a development environment running.

### Start with Docker Compose

To start the services defined in your `docker-compose.yml`, run the following command:

```bash
docker-compose up -d
```

This command will start all services in the background.


#### Download ollama LLM model
After docker-compose complete

```bash
docker exec -it ollama bash
```

#### Download ollama model

```bash
ollama pull llama2
```

### Setting Up the Python Environment

It's recommended to use pipenv for managing the project's dependencies. If you haven't installed pipenv yet, you can install it by running:

```bash
pip install pipenv
```

After installing pipenv, you can set up the project's virtual environment and install the dependencies as follows:

```bash
cd /path/to/your/project
pipenv install
```

### Running the Application

Once the setup is complete, you can run the Python script using pipenv to ensure it executes within the virtual environment:

enter virtual environment

```bash
pipenv shell
```

execute text_embedding change file path you want to test

```bash
python text_embedding.py
```

execute main

```bash
python main.py
```
