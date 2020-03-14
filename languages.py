import fasttext
import sys, os

bundle_dir = './'

if hasattr(sys, '_MEIPASS'):
    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

class LanguageChecker:
    def __init__(self):
        pretrained_model_path = os.path.join(bundle_dir, 'models/languages.ftz')
        self.model = fasttext.load_model(pretrained_model_path)

    def predict_languages(self, text):
        this_D = {}
        fla = text.split('\n')
        fla = [line.strip('\n').strip(' ') for line in fla]
        fla = [line for line in fla if len(line) > 0]

        for line in fla:
            language_tuple = self.model.predict(line)
            prediction = language_tuple[0][0].replace('__label__', '')
            value = language_tuple[1][0]

            if prediction not in this_D.keys():
                this_D[prediction] = 0
            this_D[prediction] += value

        self.this_D = this_D

    def determine_lang(self, text):
        self.predict_languages(text)

        max_value = max(self.this_D.values())
        sum_of_values = sum(self.this_D.values())
        confidence = max_value / sum_of_values
        max_key = [key for key in self.this_D.keys()
                   if self.this_D[key] == max_value][0]

        # language code
        return max_key
