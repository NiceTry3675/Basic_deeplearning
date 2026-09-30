"""
보고서 md → docx 변환 (표지 1쪽 + 목차 1쪽 + 본문)

  python build_report.py

1) md에서 '본문 시작' 표시 아래만 pandoc으로 docx 변환 (## → 제목 1, ### → 제목 2)
2) docx XML을 후처리하여 표지, 목차(Word TOC 필드), 쪽 번호 바닥글, A4 용지, 한글 글꼴(맑은 고딕) 적용
3) LibreOffice가 있으면 PDF로 렌더링해 각 제목의 쪽 번호를 찾아 목차에 채움
   (Word에서 열면 목차에서 우클릭 → '필드 업데이트'로 다시 계산 가능)
"""
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from xml.sax.saxutils import escape

import pypandoc

HERE    = os.path.dirname(os.path.abspath(__file__))
SRC     = os.path.join(HERE, "202258096_임준현_딥러닝기초_과제3보고서.md")
OUT     = SRC.replace(".md", ".docx")
MARKER  = "<!-- 본문 시작"
FONT_KO = "Malgun Gothic"

COVER = {
    "course":   "딥러닝기초",
    "title":    "과제 3 보고서",
    "subtitle": "MNIST 원본 크기(28×28) 입력을 사용한\nCNN 학습·성능평가·추론",
    "info":     [("학번", "202258096"), ("이름", "임준현"), ("제출일", "2026년 9월 29일")],
}

# A4, 여백 2.54cm (twip 단위)
PAGE_W, PAGE_H, MARGIN = 11906, 16838, 1440
TEXT_W = PAGE_W - 2 * MARGIN

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
R_NS = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'

#%% XML 조각
def run(text, bold=False, size=None, color=None):
    rpr = ""
    if bold:  rpr += "<w:b/><w:bCs/>"
    if color: rpr += f'<w:color w:val="{color}"/>'
    if size:  rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

def para(content, style=None, jc=None, before=None, after=None, extra_ppr=""):
    ppr = ""
    if style: ppr += f'<w:pStyle w:val="{style}"/>'
    ppr += extra_ppr
    if before is not None or after is not None:
        ppr += f'<w:spacing w:before="{before or 0}" w:after="{after or 0}"/>'
    if jc: ppr += f'<w:jc w:val="{jc}"/>'
    return f"<w:p><w:pPr>{ppr}</w:pPr>{content}</w:p>"

PAGE_BREAK = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

def cover_xml():
    c = COVER
    rule = ('<w:pBdr><w:bottom w:val="single" w:sz="12" w:space="12" w:color="1F4E79"/></w:pBdr>')
    xml  = para(run(c["course"], size=32, color="595959"), jc="center", before=3600, after=240)
    xml += para(run(c["title"], bold=True, size=64, color="1F4E79"), jc="center", after=360, extra_ppr=rule)
    for i, line in enumerate(c["subtitle"].split("\n")):
        xml += para(run(line, size=32), jc="center", before=360 if i == 0 else 0, after=80)
    # 학번/이름/제출일: 테두리 없는 2열 표로 정렬
    rows = ""
    indent = '<w:ind w:left="360"/>'
    for k, v in c["info"]:
        rows += ("<w:tr>"
                 f'<w:tc><w:tcPr><w:tcW w:w="1500" w:type="dxa"/></w:tcPr>{para(run(k, bold=True, size=26), jc="distribute", after=120)}</w:tc>'
                 f'<w:tc><w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>{para(run(v, size=26), after=120, extra_ppr=indent)}</w:tc>'
                 "</w:tr>")
    xml += para("", before=4200)
    xml += ('<w:tbl><w:tblPr><w:tblW w:w="4500" w:type="dxa"/><w:jc w:val="center"/>'
            '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/>'
            '<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
            '<w:tblLayout w:type="fixed"/></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="1500"/><w:gridCol w:w="3000"/></w:tblGrid>' + rows + "</w:tbl>")
    return xml + PAGE_BREAK

def toc_xml(entries, pages):
    """entries: [(level, bookmark, text)], pages: {bookmark: page}"""
    tab = f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_W}"/></w:tabs>'
    body = ""
    for i, (lvl, bm, text) in enumerate(entries):
        field_begin = ('<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
                       '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
                       '<w:r><w:fldChar w:fldCharType="separate"/></w:r>') if i == 0 else ""
        field_end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>' if i == len(entries) - 1 else ""
        page = str(pages.get(bm, ""))
        link = (f'<w:hyperlink w:anchor="{bm}" w:history="1">'
                f'{run(text, bold=(lvl == 1))}<w:r><w:tab/></w:r>'
                '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
                f'<w:r><w:instrText xml:space="preserve"> PAGEREF {bm} \\h </w:instrText></w:r>'
                '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
                f'{run(page, bold=(lvl == 1))}'
                '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:hyperlink>')
        body += para(field_begin + link + field_end, style=f"TOC{lvl}", extra_ppr=tab)
    heading = para(run("목차"), style="TOCHeading")
    return ('<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents"/>'
            f'<w:docPartUnique/></w:docPartObj></w:sdtPr><w:sdtContent>{heading}{body}'
            "</w:sdtContent></w:sdt>" + PAGE_BREAK)

FOOTER_XML = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {W_NS} {R_NS}>'
              '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
              '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
              '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r>'
              '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')

BLOCK_PPR = ('<w:pBdr><w:left w:val="single" w:sz="24" w:space="8" w:color="2A78D6"/></w:pBdr>'
             '<w:shd w:val="clear" w:color="auto" w:fill="EEF4FB"/>'
             '<w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/>'
             '<w:ind w:left="240" w:right="120" w:firstLine="0"/>')

EXTRA_STYLES = f"""
<w:style w:type="paragraph" w:styleId="TOC1"><w:name w:val="toc 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
  <w:uiPriority w:val="39"/><w:unhideWhenUsed/><w:pPr><w:spacing w:before="100" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>
  <w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TOC2"><w:name w:val="toc 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
  <w:uiPriority w:val="39"/><w:unhideWhenUsed/><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:left="440"/></w:pPr>
  <w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>
"""

#%% docx 후처리
def headings(doc):
    """pandoc 출력에서 (level, bookmark, text) 목록 추출"""
    out = []
    pat = re.compile(r'<w:bookmarkStart w:id="\d+" w:name="([^"]+)" />\s*'
                     r'<w:p><w:pPr><w:pStyle w:val="Heading([12])" />.*?</w:p>', re.S)
    for m in pat.finditer(doc):
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", m.group(0)))
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        out.append((int(m.group(2)), m.group(1), text))
    return out

def postprocess(src_docx, dst_docx, pages):
    tmp = dst_docx + ".tmp"
    with zipfile.ZipFile(src_docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        entries = None
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                doc = data.decode()
                entries = headings(doc)
                body_start = doc.index("<w:body>") + len("<w:body>")
                doc = doc[:body_start] + cover_xml() + toc_xml(entries, pages) + doc[body_start:]
                # A4 용지, 여백, 바닥글(쪽 번호), 표지는 쪽 번호 숨김(titlePg)
                sect = (f'<w:footerReference w:type="default" r:id="rIdFooter1"/>'
                        f'<w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
                        f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" '
                        'w:header="720" w:footer="720" w:gutter="0"/><w:titlePg/>')
                doc = doc.replace("<w:sectPr>", "<w:sectPr>" + sect, 1)
                data = doc.encode()
            elif item.filename == "word/styles.xml":
                st = data.decode()
                # 한글 글꼴 명시 (테마의 동아시아 글꼴 대신 맑은 고딕), 언어 한국어
                st = re.sub(r'w:eastAsiaTheme="(minor|major)EastAsia"', f'w:eastAsia="{FONT_KO}"', st)
                st = st.replace('w:eastAsia="zh-CN"', 'w:eastAsia="ko-KR"')
                # 인용 블록(결과 요약·알아두기 상자): 옅은 배경 + 왼쪽 강조선
                st = re.sub(r'(<w:style [^>]*w:styleId="BlockText".*?<w:pPr>).*?(</w:pPr>)',
                            r'\1' + BLOCK_PPR + r'\2', st, count=1, flags=re.S)
                st = st.replace("</w:styles>", EXTRA_STYLES + "</w:styles>")
                data = st.encode()
            elif item.filename == "word/_rels/document.xml.rels":
                data = data.decode().replace(
                    "</Relationships>",
                    '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                    'relationships/footer" Target="footer1.xml"/></Relationships>').encode()
            elif item.filename == "[Content_Types].xml":
                data = data.decode().replace(
                    "</Types>",
                    '<Override PartName="/word/footer1.xml" ContentType="application/'
                    'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>').encode()
            zout.writestr(item, data)
        zout.writestr("word/footer1.xml", FOOTER_XML)
    os.replace(tmp, dst_docx)
    return entries

#%% 쪽 번호 계산 (LibreOffice 렌더링)
def find_pages(docx, entries):
    soffice = shutil.which("soffice")
    if not soffice or not shutil.which("pdftotext"):
        print("LibreOffice/pdftotext 없음: 목차 쪽 번호는 Word에서 '필드 업데이트'로 채워야 함")
        return {}
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        # WSL: Windows의 맑은 고딕으로 렌더링해야 Word와 줄바꿈·쪽 나눔이 비슷해짐
        win_fonts = "/mnt/c/Windows/Fonts"
        if os.path.exists(os.path.join(win_fonts, "malgun.ttf")):
            os.makedirs(os.path.join(td, "fonts"))
            for f in ("malgun.ttf", "malgunbd.ttf"):
                if os.path.exists(os.path.join(win_fonts, f)):
                    shutil.copy(os.path.join(win_fonts, f), os.path.join(td, "fonts"))
            conf = os.path.join(td, "fonts.conf")
            with open(conf, "w") as fp:
                fp.write('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
                         '<include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
                         f'<dir>{td}/fonts</dir><cachedir>{td}/cache</cachedir></fontconfig>')
            env["FONTCONFIG_FILE"] = conf
        subprocess.run([soffice, "--headless", f"-env:UserInstallation=file://{td}/lo",
                        "--convert-to", "pdf", "--outdir", td, docx],
                       env=env, capture_output=True, timeout=300, check=True)
        pdf = os.path.join(td, os.path.basename(docx).replace(".docx", ".pdf"))
        text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
    page_texts = text.split("\f")
    norm = lambda s: re.sub(r"\s+", "", s)
    pages, start = {}, 2   # 표지(1), 목차(2) 이후부터 검색
    for _, bm, t in entries:
        key = norm(t)
        for p in range(start, len(page_texts)):
            if any(norm(line) == key for line in page_texts[p].splitlines()):
                pages[bm] = p + 1
                start = p
                break
        else:
            print("쪽 번호를 찾지 못함:", t)
    return pages

#%%
def main():
    with open(SRC, encoding="utf-8") as f:
        md = f.read()
    body = md[md.index("\n", md.index(MARKER)) + 1:]
    with tempfile.TemporaryDirectory() as td:
        body_md  = os.path.join(td, "body.md")
        raw_docx = os.path.join(td, "raw.docx")
        with open(body_md, "w", encoding="utf-8") as f:
            f.write(body)
        pypandoc.convert_file(body_md, "docx", outputfile=raw_docx,
                              extra_args=[f"--resource-path={HERE}", "--from=markdown-implicit_figures+tex_math_dollars",
                                          "--shift-heading-level-by=-1",
                                          "--metadata=lang:ko-KR"])
        # 1차: 쪽 번호 없이 생성 → 렌더링해서 쪽 번호 계산 → 2차: 쪽 번호 채워서 생성
        entries = postprocess(raw_docx, OUT, pages={})
        pages = find_pages(OUT, entries)
        postprocess(raw_docx, OUT, pages)
    print(f"saved {OUT}")
    for lvl, bm, t in entries:
        print(f"{'  ' * (lvl - 1)}{t} ... {pages.get(bm, '?')}")

if __name__ == "__main__":
    main()
