import re
import zipfile

PATH = (
    r"C:\Users\harit\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm"
    r"\LocalState\sessions\6CE0DC237D2B4A21A74FDBCD9FECA6941A014719\transfers"
    r"\2026-38\Course based project.docx"
)

with zipfile.ZipFile(PATH) as z:
    xml = z.read("word/document.xml").decode("utf-8", "ignore")

xml = re.sub(r"<w:tbl[ >]", "\n[TABLE]\n", xml)
xml = re.sub(r"</w:p>", "\n", xml)
xml = re.sub(r"</w:tr>", "\n", xml)
xml = re.sub(r"</w:tc>", " | ", xml)
xml = re.sub(r"<[^>]+>", "", xml)
print(re.sub(r"\n{3,}", "\n\n", xml).strip())
