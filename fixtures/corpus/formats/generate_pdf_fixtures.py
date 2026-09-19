"""Generate PDF fixtures with pypdf so they are reproducible from source rather than binary blobs.

Two fixtures:
  vehicle-inspection.pdf  - 4 pages, each with the SAME running header and footer, so the
                            boilerplate stripper has something real to remove, and each page
                            carries a distinct fact so page locators can be checked.
  scanned-no-text.pdf     - a page with no text layer at all, standing in for a scan. The
                            loader must extract nothing and the pipeline must reject it as
                            empty rather than index a blank document.
"""

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

OUT = Path(r"D:\projects\atlasrag\fixtures\corpus\formats")
OUT.mkdir(parents=True, exist_ok=True)

HEADER = "Meridian Logistics - Confidential"
FOOTER = "Vehicle Inspection Procedure  |  revision 7"

PAGES = [
    [
        "Vehicle Inspection Procedure",
        "Every vehicle is inspected before each shift by the driver taking it out.",
        "The inspection covers tyres, lights, brakes, mirrors and the load restraint system.",
    ],
    [
        "Tyres and brakes",
        "Tread depth below 2.4 millimetres fails the inspection and the vehicle is withdrawn.",
        "Brake travel greater than 40 millimetres is recorded as defect code BR-118.",
    ],
    [
        "Load restraint",
        "Straps showing any cut through the outer weave are replaced, never repaired.",
        "A vehicle failing load restraint is withdrawn under defect code LR-204.",
    ],
    [
        "Recording and escalation",
        "Every inspection produces a record whether or not a defect is found.",
        "Three failed inspections on one vehicle within thirty days escalate to the fleet engineer.",
    ],
]


def text_stream(lines: list[str]) -> str:
    """A minimal PDF content stream drawing each line, top to bottom."""
    out = ["BT", "/F1 11 Tf"]
    y = 760
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        out.append(f"1 0 0 1 60 {y} Tm ({escaped}) Tj")
        y -= 26
    out.append("ET")
    return "\n".join(out)


def build_text_pdf(path: Path) -> None:
    writer = PdfWriter()
    for body in PAGES:
        page = writer.add_blank_page(width=595, height=842)
        lines = [HEADER, *body, FOOTER]
        stream = DecodedStreamObject()
        stream.set_data(text_stream(lines).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")

        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = writer._add_object(font)

        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
    with path.open("wb") as handle:
        writer.write(handle)


def build_scanned_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with path.open("wb") as handle:
        writer.write(handle)


build_text_pdf(OUT / "vehicle-inspection.pdf")
build_scanned_pdf(OUT / "scanned-no-text.pdf")

# Verify the fixtures actually contain what the tests will assume.
from pypdf import PdfReader  # noqa: E402

reader = PdfReader(str(OUT / "vehicle-inspection.pdf"))
print("pages:", len(reader.pages))
for i, page in enumerate(reader.pages, start=1):
    extracted = (page.extract_text() or "").replace("\n", " | ")
    print(f"  p{i}: {extracted[:110]}")

scanned = PdfReader(str(OUT / "scanned-no-text.pdf"))
print("scanned pages:", len(scanned.pages), "text:", repr(scanned.pages[0].extract_text()))
