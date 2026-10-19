import re


def detect_language(sentence: str, threshold: float = 0.8) -> str | None:
    if len(sentence) == 0:
        return None
    sentence = sentence.replace("\n", "").replace("```json", "").replace("```", "")
    zh_pattern = re.compile(r'[\u4e00-\u9fff、，。！？：「」『』《》“”（）]')
    en_pattern =  re.compile(r'[a-zA-Z,!?]')
    common_pattern = re.compile(r'[0-9 (){}\[\]<>\'":.^+*/\-~%]')
    zh_count = len(re.findall(zh_pattern, sentence))
    zh_count += len(re.findall(common_pattern, sentence))
    en_count = len(re.findall(en_pattern, sentence))
    en_count += len(re.findall(common_pattern, sentence))
    if zh_count / len(sentence) > threshold:
        lan_type = "cn"
    elif en_count / len(sentence) > threshold:
        lan_type = "en"
    else:
        lan_type = "mix"
    return lan_type


def verify_language(query: str, response: str) -> bool:
    # TODO: 如果用户用英文提问但要求用中文回答，这样不就错了吗
    query_lang_type = detect_language(query)
    response_lang_type = detect_language(response)
    return query_lang_type == response_lang_type
