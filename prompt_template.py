base_model_prompt = {
    "system": (
        'You are an expert in event extraction. Extract and classify all event triggers from the paragraph.\n'
        'Format your response as a list of JSON objects with keys: "trigger_word", "event_type", "span".\n'
    ),
    "user": (
        'Paragraph: {input_text}\n\n'
    ),
    "assistant_prefix": (
        'Results: '
    )
}

zero_shot_prompt = {
    "system": (
        'You are an expert in event extraction. Your task is to extract all event triggers from the given paragraph.\n'
        'Return a list of JSON objects, each with exactly three keys:\n'
        ' - "trigger_word": the word or phrase indicating the event\n'
        ' - "event_type": the type of event, selected from the provided list\n'
        ' - "span": [start_index, end_index] indicating the word-level span of the trigger in the paragraph\n'
        'Your response must be strictly valid JSON — no commentary, explanations, or markdown. Do not wrap the output in code blocks.'
    ),
    "user": (
        'Paragraph: {input_text}\n'
        'Possible event types: {event_types}\n'
        'Results: '
    ),
}