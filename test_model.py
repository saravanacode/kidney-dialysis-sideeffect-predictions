from google import genai

client = genai.Client(api_key='AIzaSyDhPE5jYDn_K_jzPCm7IgtZe5xybka64Qc')

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
)

print(response.text)