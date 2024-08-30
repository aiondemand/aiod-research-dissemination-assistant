import re
from typing import Any, Dict, Union


def get_assert(
    output: str, context: Dict[str, Any]
) -> Union[bool, float, Dict[str, Any]]:
    emoticons_pattern = r"😀|😃|😄|😁|😆|😅|😂|🤣|☺️|😊|😇|🙂|🙃|😉|😌|😍|🥰|😘|😗|😙|😚|🤗|🤩|🤔|🤨|😐|😑|😶|😏|😒|😬|🙄|😯|😦|😧|😮|😲|🥱|😴|🤤|😪|😵|🤐|🥴|🤢|🤮|🤧|😷|🤒|🤕|🤑|🤠|😈|👿|👹|👺|🤡|💩|👻|💀|☠️|👽|👾|🤖|🎃|😺|😸|😹|😻|😼|😽|🙀|😿|😾|👐|🙌|👏|🤝|👍|👎|👊|✊|🤛|🤜|👈|👉|👆|👇|☝️|✋|🤚|🖐|🖖|👋|🤙|💪|🦾|🖕|✍️|🙏|💍|💄|💋|👄|👅|👂|👃|👣|👁|👀|🧠|🦴|🦷|🗣|👤|👥|🧥|👚|👕|👖|👔|👗|👙|👘|🥻|🩱|🩲|🩳|👞|👟|🥾|🥿|👠|👡|👢|👑|👒|🎩|🎓|🧢|⛑|📿|💄|🌂|☂️|🤖|🖥️|💻|🖱️|🖨️|🖲️|🕹️|🗜️|🧠|🧬|📊|📈|📉|📚|📖|🧮|🔬|🔭|📡|📱|📲|📶|🌐|🔗|⛓️"

    if re.search(emoticons_pattern, output):
        return {"pass": False, "score": 0, "reason": "Emoticons found in the output"}
    else:
        return {"pass": True, "score": 1, "reason": "No emoticons found in the output"}
