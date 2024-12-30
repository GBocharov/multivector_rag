def format_prompt(query: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": f"Перед тобой смешная картинка с текстом: {query}",
                },
            ],
        }
    ]