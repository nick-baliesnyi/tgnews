import string
import re
from nltk.stem import SnowballStemmer

stop_words_list_en = ["until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
                 "above", "below", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                 "then", "once", "here", "there", "when", "where", "all", "the", "any", "an", "a", "both", "each", "few", "more",
                 "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
                 "will", "just", "don", "should", "having"]


stop_words_list_ru = ["и", "в","во","не","что","он","на","с","со","как","а","то","все","она","так","его","но","да",
                "к","у","же","вы","за","бы","по","только","ее","было","вот","от","еще","нет","о","из","ему","когда",
                "даже","ну","вдруг","ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять","уж","вам","ведь","там",
                "потом","себя","ничего","ей","может","они","тут","где","есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб",
                "без","будто","чего","раз","тоже","себе","под","будет","ж","тогда","кто","этот","того","потому","этого","какой","совсем",
                "ним","здесь","этом","один","почти","тем","чтобы","нее","сейчас","были","куда","зачем","всех","никогда","можно","при",
                "наконец","два","об","другой","хоть","после","над","больше","тот","через","эти","нас","про","всего","них","какая","много",
                "разве","три","эту","впрочем","хорошо","свою","этой","перед","иногда","лучше","чуть","том","нельзя","такой","им",
                "более","всегда","конечно","всю","между"]


translator = str.maketrans(dict.fromkeys(string.punctuation))


stop_words = {
    'ru': set(stop_words_list_ru),
    'en': set(stop_words_list_en)
}


stemmer = {
    'ru': SnowballStemmer('russian'),
    'en': SnowballStemmer('english')
}


def remove_punctuation(text):
    text = text.replace('-', ' ') \
                .replace('«', '') \
                .replace('»', '') \
                .replace('’', '') \
                .replace('‘', '') \
                .replace('“', '') \
                .replace('”', '')
    return text.translate(translator)

def get_meta_tag_content(html, property):
    index = html.find('property="' + property + '"')

    opening_tag = html.rfind('<meta', 0, index)
    closing_tag = html.find('>', opening_tag)

    content_attr_start = html.find('content="', opening_tag, closing_tag)
    content_start = content_attr_start + len('content="')
    content_end = html.find('"', content_start, closing_tag)

    if index < 0 or opening_tag < 0 or closing_tag < 0 or content_attr_start < 0:
        return ''

    content = html[content_start:content_end]
    return content


def clean_html(text):
    output = ''
    tags = []
    tag_opened = None
    tag_closed = None
    for i in range(len(text)):
        char = text[i]
        if char == '<':
            tag_opened = i
            continue
        if char == '>':
            if not tag_opened:
                continue
            tag_closed = i
            tags.append((tag_opened, tag_closed))
            tag_opened = tag_closed = None
            continue
    for i in range(len(tags) - 1):
        end = tags[i][1]
        start = tags[i+1][0]
        output = output + text[end+1:start]

    return output or text


def tokenize(text):
    text = tokenize_to_str(text)
    return text.split(' ')


def tokenize_to_str(text):
    text = remove_punctuation(text)
    text = ' '.join(text.split())
    return text.lower()

def count_words(text):
    return len(tokenize(text))


def prepare_text_for_lang(text):
    text = clean_html(text)
    return tokenize_to_str(text)


def cutoff_words(text, cut_off):
    return ' '.join(tokenize(text)[:cut_off])


# full processing
def process_text(text, lang, max_words=100):
    word_list = tokenize(text)

    # filter stop words
    word_list = [word for word in word_list if word not in stop_words[lang]]

    # stemming
    word_list = [stemmer[lang].stem(word) for word in word_list]

    # cut off words
    text = ' '.join(word_list[:max_words])
    return text