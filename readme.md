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

### Setting Up the SQLite Database

To set up the SQLite database for the project, follow these steps:

#### Create the Database File

First, ensure that the directory for the SQLite database exists. If it doesn't, create it:

```bash
mkdir -p sqlite
```

Then, create a new SQLite database file within this directory. You can do this by running:

```bash
touch sqlite/sqlite.db
```

#### Initialize the Database Schema

To set up your database schema, you will need a .sql file with the necessary SQL commands. Assuming you have this file in the sqlite directory and it's named initialize.sql, you can execute it against your newly created database with the following command:

```bash
sqlite3 sqlite/mydatabase.db < sqlite/initialize.sql
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

execute main

```bash
python main.py
```
