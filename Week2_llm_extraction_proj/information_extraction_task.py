from groq import Groq


# Connect to Groq API using my key
client = Groq(api_key="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")

# 5 example sentences to extract info from
samples = [
    "Apple CEO Steve Jobs announced a new iPhone 16 on 9/9/2024 in USA.",
    "Elon Musk's Tesla reported $25 billion in revenueon October 16th, 2026 in Austin, Texas.",
    "Egyptian footballer Mohamed Salah scored 2 goals against Manchester City on December 1st.",
    "NASA launched the Artemis II mission on March 15th, 2025 from Kennedy Space Center in Florida.",
    "Amazon founder Jeff Bezos was in Canada last Monday.",
]

# function that sends the text to the AI and extracts the info
def extract(text):
    response = client.chat.completions.create( # dee el function that sends the chat to the ai
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": """You are an information extraction assistant.
Extract the following fields from the text:
- Person
- Organization
- Location
- Date
- Event
Reply in this EXACT format:
Person: ...
Organization: ...
Location: ...
Date: ...
Event: ...
If a field is not found, write N/A."""}, # instructions to the ai
            {"role": "user", "content": f"Text: {text}"} # the actual sentence we send
        ]
    )
    return response.choices[0].message.content.strip() # get the first response only

print("=" * 60)
print("INFORMATION EXTRACTION — 5 SAMPLE TESTS")
print("=" * 60)
print("")

for i in range(len(samples)): # loop iterating over every text one by one
    text = samples[i]
    result = extract(text) # send to AI and get facts back
    print(f"Sample {i+1}: {text}")
    print("-" * 40)
    print(result)
    print()

print("=" * 60)