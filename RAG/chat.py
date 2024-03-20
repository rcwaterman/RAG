from ollama import chat


messages = [
  {
    'role': 'user',
    'content': 'How can I embed data into mistral ai models with langchain?',
  },
]

response = chat('mistral', messages=messages, stream=True)
data = ''
dataset = []
for idx, token in enumerate(response):
    data = data + token['message']['content']
    if token['message']['content'] == '\n':
        dataset.append(data)
        print(dataset[-1])
        data = ''
        