import logging
import re
import tempfile
from difflib import SequenceMatcher

import click
import cv2
import nltk
import numpy as np
import pytesseract
import torch
from enchant.checker import SpellChecker
from pdf2image import convert_from_bytes
from PIL import Image
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer

nltk.download('punkt', quiet=True)
nltk.download('words', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ITERS = 3
IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


def get_images(path):
    if path[-3:] == 'pdf':
        pages_ = convert_from_bytes(open(path, 'rb').read())
        pages_ = [np.asarray(page) for page in pages_]
    else:
        pages_ = [cv2.imread(path)]
    return pages_


def process_image_for_ocr(img):
    new_path = set_image_dpi(img)
    new_img = cv2.imread(new_path, 0)
    im_new = remove_noise_and_smooth(new_img)
    return im_new


def set_image_dpi(im):
    im = Image.fromarray(im)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def image_smoothing(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothing(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def generate_text(pages_):
    custom_config = r'-l eng --oem 3 --psm 6'
    text_ = ''
    for img in pages_:
        processed_image = process_image_for_ocr(img)
        text_ += pytesseract.image_to_string(
            processed_image,
            config=custom_config) + '\n\n\n --- Page Ended --- \n\n\n'
    return text_


# cleanup text


def get_personslist(text_):
    personslist = []
    for sent in nltk.sent_tokenize(text_):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                personslist.insert(0, (chunk.leaves()[0][0]))
    return list(set(personslist))


def incorrect_words(text_, words):
    # using enchant.checker.SpellChecker, identify incorrect words
    d = SpellChecker("en_US")
    rep = {
        '\\': ' ',
        '\"': '"',
        '-': ' ',
        '"': ' " ',
        ',': ' , ',
        '.': ' . ',
        '!': ' ! ',
        '?': ' ? ',
        "n't": " not",
        "'ll": " will",
        '*': ' * ',
        '(': ' ( ',
        ')': ' ) ',
        "s'": "s '"
    }

    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    text_ = pattern.sub(lambda m: rep[re.escape(m.group(0))], text_)
    personslist = get_personslist(text_)
    ignorewords = personslist + [
        "\n", "\t", "!", ",", ".", "\"", "?", '(', ')', '*', "'", "--", "+",
        ":", ";", "&", "/", "$", "%", "---", "-"
    ]
    incorrectwords = [
        w for w in words if not d.check(w) and w not in ignorewords
    ]
    return incorrectwords


def suggested_words(incorrectwords):
    d = SpellChecker("en_US")
    suggestedwords_ = [d.suggest(w) for w in incorrectwords]
    return suggestedwords_


def clean_text(text_, text_or):
    words = text_.split()
    incorrectwords = incorrect_words(text_, words)
    suggestedwords_ = suggested_words(incorrectwords)

    # replace incorrect words with [MASK]

    for w in incorrectwords:

        if not re.match(r'[@_!#$%^&*()<>?/\|}{~:+-.0123456789]', w):
            to_rep = r"(\b)" + re.escape(w) + r"(\b|_)"
            text_ = re.sub(to_rep, '[MASK]', text_)
            text_or = re.sub(to_rep, '[MASK]', text_or)
        else:
            text_ = text_.replace(w, '[MASK]')
            text_or = text_or.replace(w, '[MASK]')

    new_words = text_.split()
    split_text = [
        ' '.join(new_words[i:i + 400]) for i in range(0, len(new_words), 400)
    ]

    return text_, text_or, split_text, suggestedwords_


# Load, train and predict using pre-trained model


def generate_ids(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = tokenized_text[:512]

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)
    return tokenizer, indexed_tokens, MASKIDS, segments_ids


def get_predictions(text):
    tokenizer, indexed_tokens, MASKIDS, segments_ids = generate_ids(text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    return predictions, MASKIDS, tokenizer


# Refine prediction by matching with proposals from SpellChecker


def predict_word(text_original, predictions, MASKIDS, tokenizer,
                 suggestedwords):

    for i in range(len(MASKIDS)):
        preds = torch.topk(predictions[0, MASKIDS[i]], k=50)
        indices = preds.indices.tolist()
        list1 = tokenizer.convert_ids_to_tokens(indices)
        list2 = suggestedwords[i]
        simmax = 0
        predicted_token = ''

        for word1 in list1:
            for word2 in list2:
                s = SequenceMatcher(None, word1, word2).ratio()
                if s is not None and s > simmax:
                    simmax = s
                    predicted_token = word1

        text_original = text_original.replace('[MASK]', predicted_token, 1)
    return text_original


if __name__ == '__main__':

    @click.command()
    @click.option('--input', default='test_scan', help='Input file path')
    @click.option('--output', default='output.txt', help='Output file name')
    @click.option('--verbose', is_flag=True, help='Detailed Output logs')

    # Run the pipeline and log at key moments.
    def run_pipeline(input, output, verbose):

        logging.basicConfig(
            format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%d-%m-%Y:%H:%M:%S',
            level=logging.INFO)

        if not verbose:
            logging.disable(logging.INFO)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.CRITICAL)

        file_path = input
        logger.critical("Reading data..")
        pages = get_images(file_path)

        logger.critical("Extracting text..")
        text = generate_text(pages)
        text_original = str(text)

        logger.critical("Cleaning text..")
        text, text_original, text_list, suggestedwords = clean_text(
            text, text_original)

        logger.critical("Correcting text..")
        for x in range(ITERS):
            for txt in text_list:
                predictions, mask_ids, tokenizer = get_predictions(txt)
                text_original = predict_word(text, predictions, mask_ids,
                                             tokenizer, suggestedwords)
                text = text_original

        logger.critical("Writing result to the output file..")
        with open(output, mode='w') as f:
            f.write(text_original)

        logger.critical("Finished!")

    run_pipeline()
