from openai import OpenAI


"""
When using  Chatgpt API uncomment this code
"""
#deepseek_client = OpenAI(
  #  api_key " " 
   # )

"""
For deepseek use this:
"""
deepseek_client = OpenAI(
    api_key=" ", base_url="   ")

with open("hateful-words/hate_lexicon.txt", "r") as file:
    hate_words = [line.strip() for line in file if line.strip()]


prompt = (
    "We are going to perform a Somali-English code-mixed hate speech classification task."
    " Please take note of the following words as hateful words in Somali. "
    "Use these words as an additional knowledge in your subsequent classification work:\n\n"
    + ", ".join(hate_words)
)

response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": prompt}
    ]
)

"""
If you want see the prompt output uncomment this
"""
#print(response.choices[0].message.content)

"""
The below are the Models we used
"""
# 1. gpt-4o-mini-2024-07-18    # 2. gpt-4o-2024-08-06       # 3. gpt-3.5-turbo-0125    # 4.  deepseek-chat
