from src.task import Task


class ChatBot(Task):
    def payload(self, prompt, opts):
        return {
            "uri": "/chat/completions",
            "payload": {
                "model": self.model(),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": opts["temperature"],
            }
        }
