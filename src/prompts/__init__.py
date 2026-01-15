from src.prompts.en import en_prompts
from src.prompts.it import it_prompts
from src.prompts.prompt_schema import Prompts
from typing import Dict


prompts: Dict[str, Prompts] = {"it": it_prompts, "en": en_prompts}
