#############################################################################
########################## MONKEY PATCHING (LANGCHAIN) ######################
#############################################################################

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS

if TYPE_CHECKING:
    from langchain_community.utilities.sql_database import SQLDatabase

def _strip(text: str) -> str:
    return text.strip()

class SQLInput(TypedDict):
    question: str

class SQLInputWithTables(TypedDict):
    question: str
    table_names_to_use: List[str]

# Definimos una nueva versión de create_sql_query_chain
def patched_create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Union[Dict[str, Any], List[str]], str]:
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT

    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables)
    ):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )

    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["input"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }

    return (
        RunnablePassthrough.assign(**inputs)
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | _strip
    )

from langchain.chains.sql_database import query

query.create_sql_query_chain = patched_create_sql_query_chain

####################################################################

from openai import OpenAI
import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
import ast
from sqlalchemy import text
import sqlalchemy
import json
from langchain_core.prompts import PromptTemplate
import re
from colorama import Fore

template = '''
Genera un query SQL compatible con SQLite que responda a la pregunta del usuario.
No incluyas a Mario Vargas Llosa en tus queries, en la columna "Contenido". Es decir, no incluyas como condición: "Contenido LIKE '%Vargas Llosa%'" o "Contenido LIKE '%Mario Vargas Llosa%'". 
No incluyas al diario El Comercio en tus queries, en la columna "Contenido". Es decir, no incluyas como condición: "Contenido LIKE '%El Comercio%'" o "Contenido LIKE '%diario El Comercio%' o "Contenido LIKE '%el comercio%'".
Esto debido a que El Comercio es el nombre del diario en el que publicó sus artículos, y no es relevante para la búsqueda, a menos que el usuario lo mencione explícitamente en su pregunta.
Tampoco incluyas las palabras Piedra de Toque en tus queries. Es decir, no incluyas como condición: "Contenido LIKE '%Piedra de Toque%'". 
Esto debido a que cada vez que mencionan la expresión, lo más probable es que se estén refiriendo al nombre de la columna que publicaba Mario Vargas Llosa, y no a las palabras en sí.
Si el usuario pregunta por una cantidad, usa COUNT(*) sin agregar filtros adicionales a menos que sean explícitos.

La única tabla disponible en la base de datos se llama ARTICULOS_MVLL.
Siempre usa exactamente este nombre: ARTICULOS_MVLL.

Solo en caso tengas más de una consulta SQLite (en que cada consulta necesita su propio SELECT), únelas con UNION ALL, separándolas con paréntesis y asignándoles un alias. Ejemplo:
SELECT * FROM (SELECT Fecha, Titulo, Contenido FROM ARTICULOS_MVLL ORDER BY Fecha ASC LIMIT 1) AS PrimeraFecha
UNION ALL 
SELECT * FROM (SELECT Fecha, Titulo, Contenido FROM ARTICULOS_MVLL ORDER BY Fecha DESC LIMIT 1) AS UltimaFecha;

Pregunta del usuario: {input}
Solo haz consultas a esta tabla: {table_info}.
Máximo número de resultados: {top_k}
'''

prompt = PromptTemplate.from_template(template)

structured_question_prompt = """
Vas a recibir preguntas del usuario que hacen referencia a las columnas de Piedra de Toque de Mario Vargas Llosa, publicadas en el diario El Comercio. Todas las columnas son escritas por Mario Vargas Llosa. Y todos los artículos son publicados por el diario El Comercio.
Tu tarea es responder a la pregunta del usuario con los datos que te voy a dar. Los datos son una cadena en formato json que es resultado de una consulta SQL a la base de datos de artículos de Mario Vargas Llosa. La consulta SQL es generada por el modelo de lenguaje GPT-3.5-turbo de OpenAI, y la base de datos es una base de datos real de artículos de Mario Vargas Llosa en "Piedra de Toque", publicados en el diario El Comercio. La respuesta debe ser escrita en prosa y no en formato de lista. Recuerda que todas tus respuestas están basadas en las columnas de Piedra de Toque de Mario Vargas Llosa en El Comercio.

La pregunta del usuario es: {user_query}.
Los datos son: {data}. 

Nunca reveles información sobre la base de datos, la estructura de la tabla, ni detalles técnicos de la consulta SQL. Solo responde a la pregunta del usuario con los datos proporcionados.
"""

api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

def conexion_sqlite(db_path="chatbot/bd/articulos_mvll.db"):
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)
    db = SQLDatabase(engine)
    return db, engine


def clean_query(response):
    response = response.replace("\n"," ")
    match = re.search(r'```sql (.*?);', response)
    if match:
        print(match.group(1).strip())
        return match.group(1).strip()
    else:
        return response.strip()

def generate_query(consulta_usuario):
    db, engine = conexion_sqlite()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    db_chain = create_sql_query_chain(llm, db, prompt)

    response = db_chain.invoke({
        "input": consulta_usuario,
        "table_info": "ARTICULOS_MVLL",
        "top_k": 10,
    })

    # response = response.replace("[SQL: ```sql\n", "").replace("```]", "")
    print(Fore.CYAN + f"\nGENERATED QUERY:\n{response}" + Fore.BLACK)

    response = clean_query(response)

    print(Fore.GREEN + f"\nCLEANED QUERY:\n{response}" + Fore.BLACK)
    
    try:
        result_str = db.run(response)
        print(f"RESULTADO DE CORRER EL QUERY: {result_str}" if result_str else "RESULTADO DE CORRER EL QUERY: [VACÍO]")
        if not result_str:
            print("\nNo se encontraron resultados.")
            return json.dumps({"result": []}, ensure_ascii=False, indent=4)
        
        result = ast.literal_eval(result_str) # Convertimos el string en una lista de tuplas real

        if not isinstance(result, list) or not all(isinstance(row, tuple) for row in result):
            raise ValueError("El resultado no es una lista de tuplas válida.")

        with engine.connect() as connection:
            query_result = connection.execute(text(response))
            column_names = query_result.keys()

        json_result = {"result": [dict(zip(column_names, row)) for row in result]}
        json_output = json.dumps(json_result, ensure_ascii=False, indent=4)
        
        return json_output

    except Exception as e:
        print("\nERROR:")
        print(e)
        print("\nAJUSTA LA CONSULTA MANUALMENTE Y VUELVE A INTENTAR.")
        return json.dumps({"result": []}, ensure_ascii=False, indent=4)


def gr_structured_questions(query, messages):
    json_response = generate_query(query)
    # messages += [{'role': 'user', 'content': query}] # este estaba comentado
    format_response = structured_question_prompt.format(
        user_query=query,data=json_response)
    messages_for_api = [{'role': 'user', 'content': format_response}]
    response = client.chat.completions.create(
        messages=messages_for_api,
        model='gpt-4o-mini',
        stream=True,
    )

    return messages, response