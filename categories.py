import fasttext
import sys
import os
import json

bundle_dir = './'

if hasattr(sys, '_MEIPASS'):
    bundle_dir = getattr(
        sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

LANG_CODES = ['en', 'ru']

class CategoryChecker:
    def __init__(self):
        self.models = {}
        for lang in LANG_CODES:
            self.models[lang] = fasttext.load_model(
                os.path.join(bundle_dir, 'models/categories_' + lang + '.ftz'))

    def determine_category(self, text, lang):
        label_tuple = self.models[lang].predict(text)
        prediction = label_tuple[0][0].replace('__label__', '')
        return prediction
