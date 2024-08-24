from openai import OpenAI

from commons.constants import OPENAI_API_KEY


class OpenAIGPT:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def set_model(self, model_name="gpt-3.5-turbo-1106"):
        self.model_name = model_name

    def get_response(self, prompt, temperature=0.6, is_json=False):
        if is_json:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Your key role is data annotation / data labeling.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=temperature,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        return response
