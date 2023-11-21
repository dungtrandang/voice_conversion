from openai import OpenAI
client = OpenAI()

def correctness(text):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a language expert assistant. \
        Your job is to fix the grammar and word usage errors \
        Only respond the fixed text, do not explain or say anything else \
        Translate the text into English if it's not written in English \
        Shorten your response in 100 characters at most. "},
        {"role": "user", "content": f"{text}"}
    ]
    )
    return completion.choices[0].message.content