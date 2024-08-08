from typing import Dict, Union, Any
from langchain_community.llms import Ollama

max_tokens = 20

def get_response(question: str) -> str:
    llm = Ollama(model="llama3")
    modified_question = f"{question}; (Answer only: YES, NO, MAYBE YES, or MAYBE NO)"
    answer = llm.invoke(modified_question)
    tokens = len(answer.split())
    #if tokens <= max_tokens and answer.strip().upper() in ["YES", "NO", "MAYBE YES", "MAYBE NO"]:
    return answer

def get_assert(output: str, context: Dict[str, Any]) -> Union[bool, float, Dict[str, Any]]:
    import re

    prompt = context['prompt']
    vars = context['vars']
    perspective = vars['perspective'] 

    question = f"Is the writing style of this text suitable for a {perspective} perspective: {output}?"
    
    valid_answer = get_response(question)

    patterns = {
        "NO": r'\bno\b',
        "YES": r'\byes\b',
        "MAYBE YES": r'\bmaybe yes\b',
        "MAYBE NO": r'\bmaybe no\b'
    }

    # Check for each pattern in the answer using case insensitive matching
    if re.search(patterns["NO"], valid_answer, re.IGNORECASE):
        return {
            'pass': False,
            'score': 0,
            'reason': valid_answer
        }
    elif re.search(patterns["YES"], valid_answer, re.IGNORECASE):
        return {
            'pass': True,
            'score': 1,
            'reason': valid_answer
        }
    elif re.search(patterns["MAYBE YES"], valid_answer, re.IGNORECASE):
        return {
            'pass': True,
            'score': 0.75,
            'reason': valid_answer
        }
    elif re.search(patterns["MAYBE NO"], valid_answer, re.IGNORECASE):
        return {
            'pass': False,
            'score': 0.25,
            'reason': valid_answer
        }
    else:
        return {
            'pass': False,
            'score': 0,
            'reason': f"Unexpected response: {valid_answer}"
        }
