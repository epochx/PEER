#!/usr/bin/env python
# -*-coding: utf8 -*-

import re
from ftfy import fix_text
from nltk.tokenize import PunktSentenceTokenizer
from somajo import Tokenizer, SentenceSplitter


NUMBER_REGEX = re.compile(r"[\$+-]?\d+(?:\.\d+)?")
NUMBER_TOKEN = '<NUM>'

TITLE_REGEX = re.compile(r"(=)\1{2}")
TITLE_TOKEN = ' === '

URL_REGEX = re.compile(r"(?P<url>https?://[^\s]+)")
URL_TOKEN = '<URL>'

def token_function(nl_token):

    if NUMBER_REGEX.match(nl_token):
        return NUMBER_TOKEN
    else:
        return nl_token


class WordTokenizer(object):

    def __init__(self, language='en'):
        self.language = language
        if language == 'en':
            self.tokenizer = TreebankTokenizer()
        elif language == 'de':
            self.tokenizer = Tokenizer(split_camel_case=True,
                                       token_classes=False,
                                       extra_info=False)
        else:
            raise NotImplementedError

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)


class SentenceTokenizer(object):

    def __init__(self, language='en'):
        self.language = language
        if language == 'en':
            self.tokenizer = PunktSentenceTokenizer()
        elif language == 'de':
            self.tokenizer = SentenceSplitter(is_tuple=False)
        else:
            raise NotImplementedError

    def tokenize(self, sentences):
        if self.language == 'en':
            return self.tokenizer.tokenize(sentences)
        else:
            return self.tokenizer.split(sentences)


def contains(sentence, phrase):
    """
    Check if word/phrase is inside the sentence.
    Input:
    words: string or iterable of words
    key  : function to be applied to every token in the sentence
           by default we use the lowercase of each token.string.
    Output
    If True, returns the first position of the word, otherwise
    returns None.
    """

    big = sentence

    if isinstance(phrase, str):
        small = [phrase]
    else:
        small = phrase

    if len(small) == 1:
        small = small[0]
        if small in big:
            return big.index(small), big.index(small)
        return None
    else:
        for i in range(len(big) - len(small) + 1):
            for j in range(len(small)):
                if small[j] is None:
                    pass
                elif big[i + j] != small[j]:
                    break
            else:
                return i, i + len(small)
        return None


def delete_wikigen_patterns(message):

    def is_revert(message):
        if len(message) == 1:
            if any([item in message for item in ['rv', 'rvv']]):
                return True
        else:
            concat_message = ' '.join(message).lower()
            if concat_message.startswith('revert'):
                return True
            if concat_message.startswith('undid'):
                return True
            elif any([item in concat_message for item in ['rv ', 'rvv ']]):
                return True
            else:
                return False

    if message:
        if is_revert(message):
            return []
        # for regex in regex_list:
        #     m = match(regex, message)
        #     if m:
        #         start, end = m.span()
        #         message = message[:start] + message[end:]
    return message


class TreebankTokenizer():
    """
    Penn Treebank Tokenizer
    The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.
    This implementation is a port of the tokenizer sed script written by Robert McIntyre
    and available at http://www.cis.upenn.edu/~treebank/tokenizer.sed.
    """

    #starting quotes
    STARTING_QUOTES = [
        (re.compile(r'^\"'), r'``'),
        (re.compile(r'(``)'), r' \1 '),
        (re.compile(r'([ (\[{<])"'), r'\1 `` '),
    ]

    #punctuation
    PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
        (re.compile(r'[?!]'), r' \g<0> '),

        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    #parens, brackets, etc.
    PARENS_BRACKETS = [
        (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> '),
        (re.compile(r'--'), r' -- '),
    ]

    #ending quotes
    ENDING_QUOTES = [
        (re.compile(r'"'), " '' "),
        (re.compile(r'(\S)(\'\')'), r'\1 \2 '),

        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
                     re.compile(r"(?i)\b(d)('ye)\b"),
                     re.compile(r"(?i)\b(gim)(me)\b"),
                     re.compile(r"(?i)\b(gon)(na)\b"),
                     re.compile(r"(?i)\b(got)(ta)\b"),
                     re.compile(r"(?i)\b(lem)(me)\b"),
                     re.compile(r"(?i)\b(mor)('n)\b"),
                     re.compile(r"(?i)\b(wan)(na) ")]

    CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
                     re.compile(r"(?i) ('t)(was)\b")]

    def __init__(self):
        pass

    def tokenize(self, text):

        # clean text a little bit
        text = fix_text(text)

        # replace URL
        text = re.sub(URL_REGEX, URL_TOKEN, text)

        # help tokenizer with wikitext
        text = text.replace('|', ' | ')
        text = re.sub(TITLE_REGEX, TITLE_TOKEN, text)

        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        #add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r' \1 \2 ', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r' \1 \2 ', text)

        output = text.split()

        output = [token_function(token)
                  for token in output]

        return output



class RegexSentenceTokenizer():

    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    def __init__(self):
        pass

    def tokenize(self, text):
        if "\n" in text:
            result  = []
            for chunk in text.split("\n"):
                result += self._tokenize(chunk)
        else:
            result = self._tokenize(text)
        return result

    def _tokenize(self, text):
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(self.prefixes,"\\1<prd>",text)
        text = re.sub(self.websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + self.caps + "[.] "," \\1<prd> ",text)
        text = re.sub(self.acronyms+" "+self.starters,"\\1<stop> \\2",text)
        text = re.sub(self.caps + "[.]" + self.caps + "[.]" + self.caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(self.caps + "[.]" + self.caps + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+self.suffixes+"[.] "+self.starters," \\1<stop> \\2",text)
        text = re.sub(" "+self.suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + self.caps + "[.]"," \\1<prd>",text)
        #if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]

        return sentences
