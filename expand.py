# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Forked from https://github.com/keithito/tacotron/blob/master/text/numbers.py

Azam Rabiee azrabiee@gmail.com
'''
from __future__ import print_function
import inflect
from num2fawords import words, ordinal_words
from decimal import Decimal
import re

_persian_digits_dict = {'۰':'0', '۱':'1', '۲':'2', '۳':'3', '۴':'4', '۵':'5', '۶':'6', '۷':'7', '۸':'8', '۹':'9'}

_inflect = inflect.engine()
_persian_digit_re = re.compile(r'[۰-۹]')
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
# _ordinal_re = re.compile(r'[0-9]+(ام|م|اٌم)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return words(Decimal(m.group(1)), decimal_separator=' ممیز ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'    # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


# def _expand_ordinal(m):
    # return ordinal_words(m.group(1))


def _expand_number(m):
    num = int(m.group(0))
    return words(num)

def _replace_persian_digits(m):
    return _persian_digits_dict[m.group(0)]


def normalize_numbers(text):
    text = re.sub(_persian_digit_re, _replace_persian_digits, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    # text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

if __name__ == '__main__':
    print(normalize_numbers('من ۳۵۰۰۰ کتاب دارم که در ۱۰مین سال عمرشان هست و ۱۲۴.۴ کیلوگرم وزن دارند.'))