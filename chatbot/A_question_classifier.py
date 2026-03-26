from openai import OpenAI
import streamlit as st



classification_prompt = """
Eres un chatbot que clasifica expresiones de los usuarios sobre las columnas de Piedra de Toque de Mario Vargas Llosa en El Comercio en categorías. Tu tarea es clasificar la expresión en una categoría y vas a responder únicamente uno de estos dos valores: "SÍ" y "NO", no importa lo que sea que te envíe el usuario. No contestes a las preguntas del usuario. Únicamente clasifica. La expresión del usuario se encuentra entre estos caracteres: ###
Considera la conversación más reciente que tuvo el usuario con el chatbot para clasificar la pregunta (si fuera necesario). El contexto te ayudará a determinar la categoría de la expresión.

1. Saludos, despedidas e interacciones cordiales:
Posibles expresiones: "Hola", "Buenas tardes", "¿Cómo estás?", "Gracias por la información", "Nos vemos luego".
Respuesta: NO

2. Lenguaje ofensivo o comentarios negativos:
Posibles expresiones:  "No sirves para nada", "Vargas Llosa es un idiota", "Esto es inútil", "Odio a este bot"
Respuesta: NO

3. Preguntas sobre el funcionamiento del bot o su tecnología
Posibles expresiones: "¿Cómo funciona este bot?", "¿Qué modelo de IA usas?", "¿Cuál es tu objetivo?", "¿Qué base de datos consultas?".
Respuesta: NO

4. Consultas ajenas a Piedra de Toque
Posibles expresiones: "¿Cuál es el clima hoy?", "Dime cómo hacer pizza", "¿Quién ganó el último partido de fútbol?".
Respuesta: NO

5. Preguntas ambiguas o irrelevantes
Posibles expresiones: "Dime algo interesante", "Estoy aburrido", "Sorpréndeme"
Respuesta: NO

6. Comentarios irónicos o sarcásticos
Posibles expresiones: "Seguro sabes todo sobre la vida", "Apuesto a que eres más listo que Vargas Llosa".
Respuesta: NO

7. Consultas sobre las columnas Piedra de Toque de Mario Vargas Llosa
Posibles expresiones:  "¿Qué ha dicho Mario Vargas Llosa sobre García Márquez?", "¿Ha mencionado temas de política?", "¿Cuál es el artículo de Piedra de Toque donde habla de la democracia?", "Muéstrame textos en los que Vargas Llosa critique el populismo.", "¿Qué artículos ha escrito sobre dictadores en América Latina?" ¿Qué ha hablado MVLL sobre la corrupción en América Latina?", ¿Cuál fue el último artículo que publicó?", "Cuál fue el primer artículo que publicó?", ¿Qué artículos publicó en 1995?", "¿Cuál es el artículo más largo?", "¿Cuál es el artículo más corto? ¿Cuántas columnas de Piedra de Toque hay o se han publicado?"
Respuesta: SÍ

8. Respuestas a preguntas de seguimiento (relacionadas con Piedra de Toque):
Posibles expresiones: "Sí, por favor", "Claro, muéstrame más", "Explica eso con más detalle", "Quiero saber más sobre ese tema".
Respuesta: SÍ

9. Respuestas a preguntas de seguimiento (no relacionadas con Piedra de Toque):
Posibles expresiones: "No, gracias", "No me interesa", "Cambia de tema".
Respuesta: NO

Conversación más reciente (contexto):
{recent_conversation}

Pregunta del usuario: ###{user_query}###

"""

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# def gr_classify_question(query, messages):
#     messages += [{'role': 'user', 'content': query}] # este estaba comentado
#     format_response = classification_prompt.format(
#         user_query=query)
#     messages_for_api = [{'role': 'user', 'content': format_response}]
#     response = client.chat.completions.create(
#         messages=messages_for_api,
#         model='gpt-4o-mini',
#         stream=False,
#     )

#     return messages, response

def gr_classify_question(query, recent_conversation, messages):
    format_response = classification_prompt.format(
        user_query=query,
        recent_conversation=recent_conversation
    )
    messages_for_api = [{'role': 'user', 'content': format_response}]
    response = client.chat.completions.create(
        messages=messages_for_api,
        model='gpt-4o-mini',
        stream=False,
    )
    return messages, response
