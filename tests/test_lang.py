"""The language layer: Cyrillic must survive, and English must not change at all.

Before this, every one of the splitting helpers was ASCII-only, so a Russian dictation was
tokenised to nothing: `_norm_words("привет")` was `[]`, `train_vsr._targets(...)` was `""`.
"""
import pytest

from lipflow import lang


@pytest.fixture(autouse=True)
def restore_language():
    """The active language is process state; never leak it into another test."""
    before = lang.active()
    yield
    lang.set_active(before)


def test_cyrillic_words_are_no_longer_dropped():
    from lipflow.cleanup import _norm_words
    from lipflow.corrections import _words
    from lipflow.personal import words_of
    from lipflow.visemes import visemes

    lang.set_active("ru")
    assert _norm_words("привет как дела") == ["привет", "как", "дела"]
    assert _words("Привет, мир!") == ["привет", "мир"]
    assert words_of("Привет, мир!") == ["привет", "мир"]
    assert visemes("привет") == "MWIFIT"


def test_russian_numbers_become_digits():
    lang.set_active("ru")
    assert lang.numbers_to_digits("тысяча девятьсот сорок три") == "1943"
    assert lang.numbers_to_digits("две тысячи двадцать пять") == "2025"
    assert lang.numbers_to_digits("двадцать пять") == "25"
    assert lang.numbers_to_digits("пять") == "пять"  # small numbers stay words, like English "five"


def test_russian_punctuation_and_pronouns():
    from lipflow.cleanup import basic_cleanup, fix_case

    lang.set_active("ru")
    assert basic_cleanup("КОГДА НАЧНЁТСЯ ВСТРЕЧА") == "Когда начнётся встреча?"
    # the question word can sit behind a preposition in Russian
    assert basic_cleanup("ВО СКОЛЬКО НАЧНЁТСЯ ВСТРЕЧА") == "Во сколько начнётся встреча?"
    assert basic_cleanup("ЭТО ГОТОВО") == "Это готово."
    assert fix_case("привет мир") == "Привет мир"  # "я" is not capitalised mid-sentence in Russian


def test_english_is_untouched():
    from lipflow.cleanup import basic_cleanup, fix_case, numbers_to_digits

    assert lang.active() == "en"  # English stays the default
    assert numbers_to_digits("married in nineteen fifty two") == "married in 1952"
    assert numbers_to_digits("one dog and two cats") == "one dog and two cats"
    assert basic_cleanup("WHAT TIME IS IT") == "What time is it?"
    assert fix_case("hello miguel i'm here") == "Hello miguel I'm here"


def test_training_targets_keep_cyrillic():
    """A Russian practice clip used to train on an empty target."""
    from lipflow.train_vsr import _targets

    class R:
        _tok = staticmethod(list)

    assert "".join(_targets(R(), "Привет, мир! 42")) == "ПРИВЕТ МИР"
    assert "".join(_targets(R(), "Hello, Miguel. It's done!")) == "HELLO MIGUEL IT'S DONE"


def test_phrase_normalisation_keeps_cyrillic():
    from lipflow.train_lm import normalise

    assert normalise("Привет, мир! Как дела?") == ["ПРИВЕТ МИР", "КАК ДЕЛА"]
    assert normalise("Hello, Miguel. It's done!") == ["HELLO MIGUEL", "IT'S DONE"]


def test_russian_practice_sentences(tmp_path, monkeypatch):
    import lipflow.personal as P
    from lipflow import practice as O

    monkeypatch.setattr(P, "PHRASES", str(tmp_path / "none.txt"))
    lang.set_active("ru")
    s = O.practice_sentences(24)
    assert len(s) == 24 and len(set(s)) == 24 and all(x in O.RUSSIAN for x in s)


def test_russian_practice_handles_the_setup_window_asking_for_double(tmp_path, monkeypatch):
    """The setup window asks for 2x N_SENTENCES so it can drop the ones already practised:
    a list shorter than that used to raise ValueError and kill the practice page."""
    import lipflow.personal as P
    from lipflow import practice as O

    monkeypatch.setattr(P, "PHRASES", str(tmp_path / "none.txt"))
    lang.set_active("ru")
    s = O.practice_sentences(O.N_SENTENCES * 2)
    assert len(s) == len(set(s)) >= O.N_SENTENCES
    assert all(x in O.RUSSIAN for x in s)


def test_russian_practice_list_is_long_enough():
    from lipflow import practice as O

    assert len(O.RUSSIAN) >= O.N_SENTENCES * 2, "the setup window can ask for 2x N_SENTENCES"


def test_cleanup_prompt_follows_the_language():
    from lipflow.cleanup import _user_prompt

    lang.set_active("ru")
    p = _user_prompt(["А Б"], "")
    assert p.startswith("Варианты:") and "Candidates" not in p
    assert "распознавания речи по губам" in lang.system_prompt()

    lang.set_active("en")
    assert _user_prompt(["A B"], "").startswith("Candidates:")


# -- the interface ----------------------------------------------------------------------
def test_every_interface_string_is_translated():
    missing = [k for k in lang.UI["en"] if k not in lang.UI["ru"]]
    assert not missing, f"no Russian for: {missing}"


def test_interface_placeholders_match_between_languages():
    """A key that lost its {placeholder} would silently drop the key name or the percentage."""
    import re

    fields = lambda s: set(re.findall(r"\{(\w+)", s))
    for key, en in lang.UI["en"].items():
        assert fields(en) == fields(lang.UI["ru"][key]), key


def test_t_follows_the_language_and_falls_back():
    lang.set_active("en")
    assert lang.t("menu_quit") == "Quit Lipflow"
    assert lang.t("hold_hint", key="Right Ctrl") == "Hold Right Ctrl to dictate, double-tap for hands-free"

    lang.set_active("ru")
    assert lang.t("menu_quit") == "Выйти из Lipflow"
    assert lang.t("hold_hint", key="Right Ctrl") == "Держите Right Ctrl для диктовки, двойное нажатие — без рук"
    assert lang.t("hold_hint", key="Right Ctrl", code="en").startswith("Hold")
    # an unknown key must show itself, never a crash
    assert lang.t("no_such_key") == "no_such_key"


def test_the_russian_interface_is_not_just_english():
    identical = [k for k, v in lang.UI["en"].items() if lang.UI["ru"][k] == v]
    # product names (Lipflow, Claude, Ollama, the Codex CLI) stay as they are; the rest must differ
    assert identical == ["loading_title", "cleanup_claude", "cleanup_codex", "cleanup_ollama"], identical


def test_every_cleanup_backend_has_a_label():
    """The tray builds these keys from cleanup.CHOICES: a gap would print "cleanup_ollama"."""
    from lipflow.cleanup import CHOICES

    for name in CHOICES:
        assert lang.UI["en"].get(f"cleanup_{name}"), name
        assert lang.UI["ru"].get(f"cleanup_{name}"), name


def test_no_interface_key_is_missing_from_the_catalog():
    """Every lang.t("...") in the source must exist in the catalog."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "lipflow"
    used = set()
    for p in root.rglob("*.py"):
        used |= set(re.findall(r'lang\.t\(\s*"([a-z0-9_]+)"', p.read_text(encoding="utf-8")))
    assert used, "the scan found no lang.t() calls at all - the pattern is wrong"
    assert not (used - set(lang.UI["en"])), f"raw keys would show: {sorted(used - set(lang.UI['en']))}"
