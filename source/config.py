summarizer_text_function = [
    {
        "name": "text_summary_and_title",
        "description": "Takes a text and returns a summary and title of it.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the overall summary.",
                },
                "summary": {
                    "type": "string",
                    "description": "The summary of text.",
                },
            },
            "required": ["title", "summary"],
        },
    }
]
