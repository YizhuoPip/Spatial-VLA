"""
base_prompter.py

a multi-turn prompt builder for ensuring consistent formatting for chat-based LLMs.
"""

from abc import ABC, abstractmethod
from typing import Optional

class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family, self.system_prompt = model_family, system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...

class PurePromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        """
        A pure prompt builder for formatting prompts for LLMs.

        :param model_family: The model family to use for formatting.
        :param system_prompt: The system prompt to use for formatting.
        """
        super().__init__(model_family, system_prompt)
        self.bos, self.eos = "<s>", "</s>"
        
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        #self.eos 指的是llm生成结束符

    def add_turn(self, role: str, message: str) -> str:
        """
        Add a turn to the prompt.

        :param role: The role of the turn.
        :param message: The message of the turn.
        """
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        return wrapped_message
    
    def get_potential_prompt(self, message: str) -> None:
        """
        Don't change the prompt! get a potential prompt for the given message.

        :param message: The message to get a potential prompt for. user
        """
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        return self.prompt.removeprefix(self.bos).rstrip()