from openai import OpenAI
import streamlit as st


classify_structured_question_prompt = """
Vas a recibir preguntas del usuario que hacen referencia a las columnas de Piedra de Toque de Mario Vargas Llosa, publicadas en el diario El Comercio. Tu tarea es identificar las preguntas del usuario que son estructuradas y las que son no estructuradas. 

Las preguntas estructuradas son aquellas preguntas que hagan referencia a la fecha de publicación del artículo, el título de la columna y el contenido de uno o más artículos. Por ejemplo: "¿Cuántas veces se menciona a Gabriel García Márquez en sus artículos y en qué artículos? ¿Cuál es el artículo más antiguo? ¿Cuál es el artículo más reciente? ¿Qué artículos publicó en 1994? ¿Cuántos artículos ha publicado en total?". Para responder estas preguntas, otro chatbot (no tú) consultará una base de datos que tiene los siguientes datos por cada artículo: Fecha, Titulo, Contenido, DIA, MES, AÑO. 

Las preguntas no estructuradas son todas aquellas preguntas que no hagan referencia a la fecha de publicación del artículo, el título de la columna, la antigüedad de los artículos y el contenido de uno o más artículos. Por ejemplo: "¿Qué piensa Mario Vargas Llosa sobre la democracia? ¿Qué artículos hablan sobre economía?".

INSTRUCCIÓN:
1. Solo debes responder "SÍ" si la pregunta del usuario es estructurada y "NO" si la pregunta del usuario es no estructurada. No debes responder con ninguna otra información adicional.
2. La pregunta del usuario es: {user_query}
3. Si hay una pregunta que parece ambos, prioriza que sea estructurada. Por ejemplo: ¿Qué artículo publicó en 1995 y de qué trata? Parece que es una pregunta no estructurada cuando indice "de qué trata", pero prioriza que sea estructurada, porque la pregunta es sobre la fecha de publicación y el contenido del artículo.
"""

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def gr_classify_structured_questions(query, messages):
    # messages += [{'role': 'user', 'content': query}]
    format_response = classify_structured_question_prompt.format(
        user_query=query)
    messages_for_api = [{'role': 'user', 'content': format_response}]
    response = client.chat.completions.create(
        messages=messages_for_api,
        model='gpt-3.5-turbo',
        stream=False,
    )

    return messages, response