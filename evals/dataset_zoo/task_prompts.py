class TaskPrompts:
    @staticmethod
    def get_text_task_prompt(captions: list):
        option_texts = ", ".join(
            [
                f"({chr(65+i)}) {caption}" if i < len(captions) - 1 else f"or ({chr(65+i)}) {caption}"
                for i, caption in enumerate(captions)
            ]
        )
        options = ", ".join([f"{chr(65+i)}" for i in range(len(captions))])
        prompt = (
            f"Does this image best match: {option_texts}? "
            f"Choices: {options}. "
            # "Note, you must choose one of the choices."
        )
        if isinstance(prompt, str):
            prompt = prompt.strip()

        return prompt

    @staticmethod
    def get_image_task_prompt(caption: str):
        prompt = (
            f'Which image better aligns with the description: "{caption}"? '
            f"Choices: First, Second. "
            # "The 'First' or the 'Second' image? "
            # "Note, you must choose one of the choices."
        )
        if isinstance(prompt, str):
            prompt = prompt.strip()

        return prompt
