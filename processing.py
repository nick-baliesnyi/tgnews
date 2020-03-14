import string, re

def clean_html(text):
    output = ''
    tags = []
    for i in range(len(text)):
        char = text[i]
        if char == '<':
            tag_opened = i
            continue
        if char == '>':
            tag_closed = i
            tags.append((tag_opened, tag_closed))
            continue
    for i in range(len(tags) - 1):
        end = tags[i][1]
        start = tags[i+1][0]
        output = output + text[end+1:start]

    return output or text


translator = str.maketrans(dict.fromkeys(string.punctuation))

stop_words = ["until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
              "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
              "then", "once", "here", "there", "when", "where", "all", "any", "both", "each", "few", "more",
              "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
              "will", "just", "don", "should"]


def remove_punctuation(text):
    return text.translate(translator)


def text_to_list(text):
    # if x not in stop_words]
    return [x for x in list(filter(None, remove_punctuation(text).replace('—', '').lower().strip().replace('\n', ' ').split(' ')))]

def prepare_text_for_lang(text):
    text = re.sub(r'\n+', '\n', text)
    return remove_punctuation(text).strip().lower()


def limit_text_words(text, count):
    return ' '.join(text_to_list(text)[:count])