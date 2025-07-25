import requests

def generate_text(prompt):
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": "Bearer sk-cd25f998a69644ad97f4422f3727795a"},
        json={
            "model": "deepseek-chat",  #
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
    )
    return response.json()["choices"][0]["message"]["content"]