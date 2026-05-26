import re


def preprocess_text(text):
    match = re.search(r'#G\d{2}#', text)
    if match:
        pid = match.group().replace('#', '').lower()
    else:
        pid = None

    cleaned_text = re.sub(r'#G\d{2}#', '', text).strip()

    return cleaned_text, pid
