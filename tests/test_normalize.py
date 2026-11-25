import unicodedata

from iwhisper.normalize import normalize


def test_danda_and_punctuation_removed():
    assert normalize("नमस्ते, दुनिया। कैसे हो?") == "नमस्ते दुनिया कैसे हो"
    assert normalize("આજે, હવામાન સારું છે.") == "આજે હવામાન સારું છે"


def test_matras_survive():
    # Whisper's BasicTextNormalizer would strip these vowel signs.
    assert normalize("किताब") == "किताब"
    assert normalize("ગુજરાતી") == "ગુજરાતી"


def test_native_digits_become_ascii():
    assert normalize("वर्ष २०२४") == "वर्ष 2024"
    assert normalize("વર્ષ ૨૦૨૪") == "વર્ષ 2024"


def test_nukta_forms_are_equal():
    precomposed = "क़"  # क़ as one code point
    decomposed = "क़"  # क + nukta
    assert normalize(precomposed) == normalize(decomposed)


def test_zero_width_and_whitespace():
    assert normalize("  क्‍ष   ​त्र \n") == unicodedata.normalize("NFC", "क्ष त्र")


def test_latin_lowercased():
    assert normalize("COVID-19 का टीका") == "covid 19 का टीका"
