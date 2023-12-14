import openai
from openai import OpenAI
import json
import streamlit as st
try:
    client = OpenAI(api_key=st.secrets["key"])
except:
    lient = OpenAI()

functions = [
    {
        "name": "refined_func",
        "description": "Shows the refined text and explaination for the revision of some text.",
        "parameters": {
            "type": "object",
            "properties": {
                "fixed text": {
                    "type": "string",
                    "description": "refined part of the text output."
                },
                "explaination": {
                    "type": "string",
                    "description": "explaination of the text output."
                }
            }
        }
    }
]

function2 = [
    {
        "name": "suggested_phrases",
        "description": "Suggest phrases to answer a question",
        "parameters": {
            "type": "object",
            "properties": {
                "hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phrase": { "type": "string", "description": "Suggested phrase, idiom, collocation to use to answer the question, e.g. focus on" },
                            "meaning": { "type": "string", "description": "Meaning of the phrase in Vietnamese, e.g. tập trung" },
                            "example": { "type": "string", "description": "An example of the phrase in use, e.g. On my busy day, I will focus on high-priority items first." },
                    },
                        "description":"List of suggested phrases"
                },
                     "required": ["suggest","gợi ý"],
                
            }
        }
    }
    }
]


def hint(level, question, user_question):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"You support the english learner to answer an English {question}. \
        The learner will ask an user_question - an question or an idea or an request related to the original question. \
        Respond 3 hints only - not full sentence, that is suitable with the level {level}. \
        Each hint includes a nice phrase in the format of a linguistic constructs (e.g. worn out, focus on, kick the bucket), \
        the respective meaning always in Vietnamese and an example related to the original question. \
        # If user_question is in Vietnamese, respond the meaning in Vietnamese. If user_question is in English, respond the meaning in English. \
         "},
        {"role": "user", "content": f"{user_question}"}
    ],
    functions = function2,
    function_call = {
        "name": "suggested_phrases"
    }
    )
    string = completion.choices[0].message.function_call.arguments
    output = json.loads(string)
    return output

def correctness(text):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a language expert assistant. \
        Your job is to fix the grammar and word usage errors. \
        Shorten your response in 100 characters at most, respond the fixed text and explaination of the correcness of choice of word and grammar. \
         "},
        {"role": "user", "content": f"{text}"}
    ],
    functions = functions,
    function_call = {
        "name": functions[0]["name"]
    }
    )
    string = completion.choices[0].message.function_call.arguments
    output = json.loads(string)
    return output.get("fixed text"), output.get("explaination")

# print(correctness("I have a dog."))

def question():
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an IELTS test maker. \
        respond an IELTS Speaking part 1 question only without any other comment, introduction, explaination or answer \
         "},
        {"role": "user", "content": f""}
    ])
    string = completion.choices[0].message.content
    return string
# print(question())
# print( hint("CERF A1", "What do you like to do in your free time?", "gợi ý vài ý tưởng để trả lời câu hỏi này"))