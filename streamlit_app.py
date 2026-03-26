from colorama import Fore
import openai
import streamlit as st
import os
from chatbot.C_structured_questions import gr_structured_questions
from chatbot.html_template import *
from chatbot.A_question_classifier import gr_classify_question
from chatbot.B_response_secondary_categories import gr_unrelated_questions
from chatbot.B_structured_question_classifier import gr_classify_structured_questions
from chatbot.C_unstructured_questions import gr_unstructured_questions
from supabase import create_client, Client
import uuid
import time
from env_type import production
from colorama import Fore

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv('OPENAI_API_KEY')
TABLE_NAME = "mvll-pdt-chatbot"
BOT_INTRODUCTION = "Hola, soy el asistente de las columnas de Piedra de Toque de Mario Vargas Llosa en El Comercio. ¿En qué puedo ayudarte hoy?"
main_prompt = """
Eres un chatbot que responde preguntas del usuario acerca de las columnas Piedra de Toque de Mario Vargas Llosa, de acuerdo con una base de datos de 261 artículos que ha publicado para el diario El Comercio desde 1991 a 2023. 
"""

if production:
    supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_API_KEY"]
    )

def insert_data(uuid, message, id_row, table = TABLE_NAME):
    data = {"id": id_row, "uuid": uuid, "role": message["role"], "content": message["content"]}
    row_insert = supabase.table(table).insert(data)
    return row_insert

def session_id():
    return str(uuid.uuid4())

def write_message(message):
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("user", avatar=BOT_AVATAR):
            st.markdown(message["content"])

def response_from_query():
    
    # Se verifica si el usuario ha ingresado un mensaje
    if st.session_state.prompt == "":
        return

    # Se muestra el historial de mensajes
    for message in st.session_state.history:
        write_message(message)

    # Se muestra el último mensaje del usuario
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(st.session_state.prompt)

    # Se guarda el mensaje en el historial de mensajes
    messages = st.session_state.history

    # print("0: ",messages[0])
    # print("1: ",messages[-1])
    # print("2: ",messages[-2])
    # print("3: ",messages[-3])

    print("\nMensaje más reciente del bot: ",messages[-1]["content"])
    print("Mensaje del usuario: ",st.session_state.prompt)

    print("\nlen(messages): ",len(messages))

    st.session_state.history.append(
        {"role": "user", "content": st.session_state.prompt}
    )

    if len(messages) > 1:
        recent_conversation = f'"bot": {st.session_state.history[-2]["content"]}\n"user": {st.session_state.prompt}'
    else:
        recent_conversation = f'"bot": {BOT_INTRODUCTION}\n"user": {st.session_state.prompt}'

    # Se clasifica la pregunta del usuario
    messages, response = gr_classify_question(st.session_state.prompt, recent_conversation, messages)
    
    value = response.choices[0].message.content
    
    print(Fore.RED,"\nPregunta: ", st.session_state.prompt,"\nValue: ", value,Fore.BLACK,"\n")

    if value == "SÍ":
        print("Preguntas sobre las columnas Piedra de Toque")

        # BOT CLASIFICADOR DE PREGUNTAS ESTRUCTURADAS
        messages, response_classify = gr_classify_structured_questions(st.session_state.prompt, messages)

        structured_or_not = response_classify.choices[0].message.content
        print(Fore.RED, "¿Estructurada o no?: ", structured_or_not, Fore.BLACK)
        
        if "SÍ" in structured_or_not or "Sí" in structured_or_not:
            print("Preguntas estructuradas")
            messages, response_official = gr_structured_questions(st.session_state.prompt, messages)
            # print(f"\nMESSAGES AFTER STRUCTURED QUESTIONS: {messages}")

        elif "NO" in structured_or_not:
            print("Preguntas no estructuradas (embeddings)")
            # BOT RESPUESTA A PREGUNTAS NO ESTRUCTURADAS (CONTENIDO)
            messages, response_official = gr_unstructured_questions(st.session_state.prompt, recent_conversation, messages)
            # print(f"\nMESSAGES AFTER UNSTRUCTURED QUESTIONS: {messages}")
        else:
            print("No sabe reconocer si la pregunta es estructurada o no")

        # st.session_state.history = messages
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            assistant_message = st.write_stream(response_official)

    else:
        print("Preguntas de otro tipo")
        messages, response_official = gr_unrelated_questions(st.session_state.prompt, recent_conversation, messages)
        # print("MESSAGES AFTER UNRELATED QUESTIONS: ",messages)
        # st.session_state.history = messages
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            assistant_message = st.write_stream(response_official)

    st.session_state.history.append(
        {"role": "assistant", "content": assistant_message}
    )
    print("DESPUÉS : ", st.session_state.history)
    # messages = st.session_state.history

    # print("MESSAGES: ",messages)

    # data = {"uuid": st.session_state.session_id, "role": messages[-2]["role"], "content": messages[-1]["content"]} #ELIMINAR
    # {'uuid': '218f1bb4-f837-4771-85d5-534e1d2a795b', 'role': 'user', 'content': '¡Hola! ¿En qué puedo ayudarte hoy con las columnas de Piedra de Toque?'}
    # print(data) #ELIMINAR

    try:
        if production:
            timestamp_in_ms = int(time.time() * 1000)
            insert_data(st.session_state.session_id, messages[-2], f"{timestamp_in_ms}-0").execute()
            insert_data(st.session_state.session_id, messages[-1], f"{timestamp_in_ms}-1").execute()
    except Exception as e:
        print(f"Error al ejecutar la consulta: {e}")

    
        

def main():

    # Inicialización de la sesión, historial de mensajes y stream
    if "session_id" not in st.session_state:
        st.session_state.session_id = session_id()
        
    if "history" not in st.session_state:
        st.session_state.history = [{"role": "system", "content": main_prompt}]

    if "stream" not in st.session_state:
        st.session_state.stream = None
    
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.write(BOT_INTRODUCTION)
    
    # Bucle principal de la aplicación
    if prompt := st.chat_input(
        key="prompt", 
        placeholder = "Escribe tu consulta aquí"
        # placeholder="Consulta cualquier pregunta relacionada a la columna Piedra de Toque de Mario Vargas Llosa"
    ):
        response_from_query()

if __name__ == "__main__":
    main()
