import logging
import re
import requests
from universe_builder.llm_classes.llm_call import LLMCall

log = logging.getLogger(__name__)


class OllamaCall(LLMCall):
    """
    NOTE: Local LLMs have limited context so it doesn't really work well.
    Better way would be to split calls and add them manually to the Class
    """

    def __init__(
        self,
        atom_type,
        atom_class,
        model_version,
        prompt_folder,
        prompt_subject,
        prompt_start_text,
        base_system_text,
        example_json_file,
        max_tokens=None,
        operation=None,
        context=None,
        effort=None,
        task_budget=None,
    ):
        # `operation`/`context` are accepted for the `llm_call_mapping`
        # contract and unused: this class builds its prompt from
        # `base_system_text` itself. `effort` (v0.6.0) likewise: a local
        # model has none to set.
        log.warning("USING LOCAL LLM")
        if effort:
            log.warning("effort %r ignored: the local model takes none", effort)
        if task_budget:
            log.warning("task budget %r ignored: the local model takes none", task_budget)
        super().__init__(
            atom_type,
            atom_class,
            model_version,
            prompt_folder,
            prompt_subject,
            prompt_start_text,
            base_system_text,
            example_json_file,
            max_tokens=max_tokens,
        )

        # One example, the first configured: this class builds its own prompt.
        first_file = self.example_json_files[0] if self.example_json_files else ""
        self.example_json = self.example_jsons[0] if self.example_jsons else None
        pretty_file_subject = re.sub(r"\.json$", "", first_file.lower().replace("_", " "))

        self.prompt_text = f"{self.prompt_start_text} '{self.prompt_subject}'"
        self.example_text = f"{self.prompt_start_text} '{pretty_file_subject}'"
        # self.system_text = self.base_system_text +
        #   "\nThis is a blank json file to fill in for the response you will give me\n" +
        #   self.blank_json
        self.system_text = self.base_system_text

        # This should be the last thing to run since it require everything else ready
        self.llm_response_json = self.call_llm_api()

    def call_llm_api(self):
        log.debug(f"Calling OpenAI API with model {self.model_version}")
        self.log_request()

        example = (f"This is an example for {self.prompt_subject} response: {self.example_json}\n"
                   if self.example_json else "")
        payload = {
            "model": self.model_version,
            "messages": [
                {
                    "role": "user",
                    "content": f"{example}{self.system_text}\n{self.prompt_text}",
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1,
                        **({"num_predict": self.max_tokens} if self.max_tokens else {})},
        }

        # A local model on a laptop can take minutes; the bound is for a server that never answers.
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)

        # Print the status code
        log.info(f"Status Code: {response.status_code}")

        # Try to parse the response as JSON
        response_json = response.json()
        # log.info(json.dumps(json.loads(response_json["message"]["content"]), indent = 4))
        returned_json = response_json["message"]["content"]
        return returned_json
