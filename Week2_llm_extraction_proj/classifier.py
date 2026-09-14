from groq import Groq


# Connect to Groq API using my key (lasts 90 days)
client = Groq(api_key="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")# Connect to Groq API using my key

# 5 examples 
samples = [
    "Realmadrid defeated Barcelona in an overtime last night.",
    "The prime minister announced there's new tax reforms.",
    "Anthropic released a new model that beats any AI out there.",
    "I workout everyday for my personal health",
    "Assaad won the game in this hangout",
]
# e3mel function that connects with the api
def classify(text):
    response = client.chat.completions.create( #dee el function that sends the chat to the ai 
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": """You are a text classifier. Classify the text into EXACTLY ONE of:
Sports, Politics, Technology, Health, Entertainment.
Reply with ONLY the category name, nothing else."""},
            {"role": "user", "content": f"Text: {text}"} #text will be the category output
        ]
    )
    return response.choices[0].message.content #get the first response only from the ai, a7san 

print("=" * 60)
print("TEXT CLASSIFICATION — 5 SAMPLE TESTS")
print("=" * 60)
print("")

for i in range(len(samples)): # dee loop iterating for every text
    text = samples[i]
    label = classify(text)
    print(f"Sample {i+1}: {text}")
    print(f" Category: {label}\n")


print("=" * 60)