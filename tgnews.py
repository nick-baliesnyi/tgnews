import os
import sys
import json
import text_processing
import numpy as np
import multiprocessing
import fasttext
from functools import partial
# from tqdm import tqdm  # TODO Remove the tqdm
from sklearn.cluster import DBSCAN
from datetime import datetime

from languages import LanguageChecker
from news import NewsChecker
from categories import CategoryChecker

LANG_CODES = ['ru', 'en']

lang_checker = LanguageChecker()
news_checker = None
category_checker = None
categories_model = None


def list_files(directory):
    r = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith('.html'):
                r.append(os.path.join(root, name))
    return r


def split_file_list(file_list):
    cores_count = multiprocessing.cpu_count()
    divider = (len(file_list)) // cores_count
    file_lists = []
    for i, file in enumerate(file_list):
        list_index = min(i // divider, cores_count-1)

        if list_index >= len(file_lists):
            file_lists.append([])

        file_lists[list_index].append(file)

    return file_lists


# Multiprocessing function
def languages_process(file_list):
    dict_part = {}
    for file_path in file_list:
        try:
            with open(file_path, 'r') as f:
                html = f.read()
                text = text_processing.prepare_text_for_lang(html)
                lang = lang_checker.determine_lang(text)

                if lang in LANG_CODES:
                    if lang not in dict_part:
                        dict_part[lang] = []
                    dict_part[lang].append(os.path.basename(file_path))
        except:
            pass
    return dict_part


def languages(directory):
    articles_dict = {}
    file_list = list_files(directory)
    file_lists = split_file_list(file_list)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    dict_parts = pool.map(languages_process, file_lists)

    for dict_part in dict_parts:
        for lang in dict_part:
            if lang not in articles_dict:
                articles_dict[lang] = []
            articles_dict[lang].extend(dict_part[lang])

    output = []
    for lang in articles_dict:
        output.append({
            'lang_code': lang,
            'articles': articles_dict[lang]
        })

    return json.dumps(output)


# Multiprocessing function
def news_process(file_list):
    part_news = []
    for file_path in file_list:
        try:
            with open(file_path, 'r') as f:
                html = f.read()
                lang_text = text_processing.prepare_text_for_lang(html)
                lang = lang_checker.determine_lang(lang_text)

                if lang in LANG_CODES:
                    text = text_processing.process_text(lang_text, lang)
                    is_news = news_checker.determine_is_news(text, lang)

                    if is_news:
                        part_news.append(os.path.basename(file_path))
        except:
            pass

    return part_news


def news(directory):
    news_dict = {'articles': []}
    file_list = list_files(directory)
    file_lists = split_file_list(file_list)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    parts_news = pool.map(news_process, file_lists)

    for part in parts_news:
        news_dict['articles'].extend(part)

    return json.dumps(news_dict)


# Multiprocessing function
def categories_process(file_list):
    dict_categories = {}
    for file_path in file_list:
        try:
            with open(file_path, 'r') as f:
                html = f.read()
                lang_text = text_processing.prepare_text_for_lang(html)
                lang = lang_checker.determine_lang(lang_text)

                if lang in LANG_CODES:
                    text = text_processing.process_text(lang_text, lang)
                    category = category_checker.determine_category(text, lang)
                    if category not in dict_categories:
                        dict_categories[category] = []
                    dict_categories[category].append(
                        os.path.basename(file_path))
        except:
            pass

    return dict_categories


def categories(directory):
    categories_dict = {}
    file_list = list_files(directory)
    file_lists = split_file_list(file_list)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    dict_parts = pool.map(categories_process, file_lists)

    for dict_part in dict_parts:
        for category in dict_part:
            if category not in categories_dict:
                categories_dict[category] = []
            categories_dict[category].extend(dict_part[category])

    output = []
    for category in categories_dict.keys():
        output.append({
            'category': category,
            'articles': categories_dict[category]
        })
    return json.dumps(output)


def threads_process(task_type, file_list):
    part_list = []
    for file_path in file_list:

        f = open(file_path, 'r')
        html = f.read()
        f.close()

        title = text_processing.get_meta_tag_content(html, 'og:title')
        published_time_str = text_processing.get_meta_tag_content(
            html, 'article:published_time')
        published_time = None
        try:
            published_time = datetime.strptime(
                published_time_str, '%Y-%m-%dT%H:%M:%S%z')
        except:
            continue

        lang_text = text_processing.prepare_text_for_lang(html)
        lang = lang_checker.determine_lang(lang_text)

        if lang in LANG_CODES:
            text = text_processing.process_text(lang_text, lang)
            model = category_checker.models[lang]
            text_vector = model.get_sentence_vector(text)
            if task_type == 'threads':
                part_list.append({'file': os.path.basename(file_path), 'title': title,
                                  'vector': text_vector, 'published_time': published_time})
            elif task_type == 'top':
                category = category_checker.determine_category(text, lang)
                part_list.append({'file': os.path.basename(file_path), 'title': title, 'vector': text_vector,
                                  'published_time': published_time, 'category': category})

    return part_list


def get_threads(task_type, directory):
    file_lists = split_file_list(list_files(directory))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    partial_func = partial(threads_process, task_type)
    part_lists = pool.map(partial_func, file_lists)

    all_articles = []
    for part_list in part_lists:
        all_articles.extend(part_list)

    # Sorting by the published time
    all_articles.sort(key=lambda x: x['published_time'], reverse=True)

    news_vectors_list = []

    for article in all_articles:
        news_vectors_list.append(article['vector'])

    # Creating numpy vector list for DBSCAN clustering algorithm
    X = np.array(news_vectors_list)

    clustering = DBSCAN(eps=0.08, min_samples=2, metric='cosine').fit(X)
    labels = list(clustering.labels_)

    threads_dict = {}

    # Mapping the DBSCAN results to the dict
    for i, label in enumerate(labels):
        if label < 0:
            continue
        if label not in threads_dict:
            threads_dict[label] = []
        if task_type == 'threads':
            threads_dict[label].append(
                {'file': all_articles[i]['file'], 'title': all_articles[i]['title']})
        elif task_type == 'top':
            threads_dict[label].append(
                {'file': all_articles[i]['file'], 'title': all_articles[i]['title'],
                 'category': all_articles[i]['category'], 'published_time': all_articles[i]['published_time']})

    return threads_dict


def threads(directory):
    threads_dict = get_threads('threads', directory)

    output = []
    # Mapping the threads dict to the required output
    for label in threads_dict:
        thread = {}
        # Newest published article title is considered to be a title
        thread['title'] = threads_dict[label][0]['title']
        thread['articles'] = [f['file'] for f in threads_dict[label]]
        output.append(thread)

    return json.dumps(output)


def top(directory):
    threads_dict = get_threads('top', directory)
    threads_with_categories = []

    # Determ the category name of a thread and find the max_inthread_count

    thread_categories_map = {}
    thread_article_count = {}
    max_inthread_count = 0

    for label in threads_dict.keys():
        categories_max_dict = {}
        count = 0
        for article in threads_dict[label]:
            if not article['category'] in categories_max_dict:
                categories_max_dict[article['category']] = 0
            categories_max_dict[article['category']] += 1
            count += 1

        # Save the articles count for each thread
        thread_article_count[label] = count

        # Find the max
        if count >= max_inthread_count:
            max_inthread_count = count

        max_category_name_count = 0

        for category in categories_max_dict:
            if categories_max_dict[category] >= max_category_name_count:
                thread_categories_map[label] = category
                max_category_name_count = categories_max_dict[category]

    for label in threads_dict.keys():
        first_date = threads_dict[label][0]['published_time']
        last_date = threads_dict[label][-1]['published_time']
        title = threads_dict[label][0]['title']
        diff = first_date - last_date
        # Count the relevance score based on the window of days and amount of articles
        relevance_score = thread_article_count[label] / max_inthread_count + diff.days * 0.1
        threads_with_categories.append(
            {'title': title, 'relevance': relevance_score, 'category': thread_categories_map[label], 'thread': threads_dict[label]})

    # Sorting by relevance
    threads_with_categories.sort(key=lambda x: x['relevance'], reverse=True)

    output = []

    threads = []
    threads_dict = {}
    for t in threads_with_categories:
        threads.append({
            'title': t['title'],
            'articles': [f['file'] for f in t['thread']]
        })
    threads_dict['any'] = threads

    for t in threads_with_categories:
        if not t['category'] in threads_dict:
            threads_dict[t['category']] = []
        
        threads_dict[t['category']].append({
            'title': t['title'],
            'articles': [f['file'] for f in t['thread']]
        })

    for t_key in threads_dict.keys():    
        output.append({
            'category': t_key,
            'threads': threads_dict[t_key]
        })
    return json.dumps(output)


def main():
    command = sys.argv[1]
    argument = sys.argv[2]
    global category_checker, news_checker
    if command == 'languages':
        print(languages(argument))
    if command == 'news':
        news_checker = NewsChecker()
        print(news(argument))
    if command == 'categories':
        category_checker = CategoryChecker()
        print(categories(argument))
    if command == 'threads':
        category_checker = CategoryChecker()
        print(threads(argument))
    if command == 'top':
        category_checker = CategoryChecker()
        print(top(argument))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
