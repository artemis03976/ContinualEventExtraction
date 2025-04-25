base_model_prompt = {
    "system": (
        'You are an expert in event extraction. Extract and classify all event triggers from the paragraph.\n'
        'Format your response as a list of JSON objects with keys: "trigger_word", "event_type", "span".\n'
    ),
    "user": (
        'Paragraph: {input_text}\n\n'
    ),
    "assistant_prefix": (
        'Results:'
    )
}