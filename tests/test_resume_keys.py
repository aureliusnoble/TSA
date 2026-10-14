"""Resume must recognise every page whose CSV was already exported.

export_transcription() only strips ".jpg" from the image name, so ".jpeg"
(and other) pages are written as "<page>.jpeg.csv". _get_processed_files()
has to normalise those names back to the key _get_pending_images() builds
from image_path.stem, otherwise a re-run reprocesses the whole input.
"""
from types import SimpleNamespace

import pandas as pd

import src.tsa_utils as tsa
from inference import Pipeline


def _bare_pipeline(input_dir, output_dir):
    self = SimpleNamespace(config=SimpleNamespace(
        directories=SimpleNamespace(input=str(input_dir), output=str(output_dir))))
    self._sanitize_filename = lambda f: Pipeline._sanitize_filename(self, f)
    self._get_processed_files = lambda: Pipeline._get_processed_files(self)
    return self


def _export(self, output_dir, image_path):
    # Same call sequence as Pipeline.run() uses after processing a page.
    tsa.PageReconstruction().export_transcription(
        pd.DataFrame({"transcription": ["x"]}), str(output_dir),
        image_path.parent.parent.name, image_path.parent.name,
        self._sanitize_filename(image_path.name))


def test_exported_pages_are_skipped_for_all_extensions(tmp_path):
    period = tmp_path / "images" / "Commune" / "1825 - 1833"
    period.mkdir(parents=True)
    names = ["Commune_1825 - 1833_page1.jpeg", "Commune_1825 - 1833_page2.jpeg",
             "page3.jpg", "page4.JPEG", "page5.png", "page6.jpg"]
    for n in names:
        (period / n).write_bytes(b"")
    output = tmp_path / "output"
    self = _bare_pipeline(tmp_path / "images", output)

    done = [period / n for n in names if n != "Commune_1825 - 1833_page2.jpeg"
            and n != "page6.jpg"]
    for p in done:
        _export(self, output, p)

    pending, skipped = Pipeline._get_pending_images(self)
    assert sorted(p.name for p in pending) == ["Commune_1825 - 1833_page2.jpeg", "page6.jpg"]
    assert skipped == len(done)


def test_nothing_skipped_without_outputs(tmp_path):
    period = tmp_path / "images" / "Commune" / "1803-1816"
    period.mkdir(parents=True)
    (period / "a.jpeg").write_bytes(b"")
    self = _bare_pipeline(tmp_path / "images", tmp_path / "output")
    pending, skipped = Pipeline._get_pending_images(self)
    assert [p.name for p in pending] == ["a.jpeg"] and skipped == 0
