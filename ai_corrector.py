from openai import OpenAI
client = OpenAI()

def correctness(text):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a language refinement assistant. \
        Refine chat for natural flow. \
        Shorten your response in 100 characters at most. "},
        {"role": "user", "content": f"Refine chat for natural flow: {text}"}
    ]
    )
    return completion.choices[0].message.content