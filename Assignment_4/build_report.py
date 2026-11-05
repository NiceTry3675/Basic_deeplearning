"""
보고서 md → docx 변환 (표지 1쪽 + 목차 1쪽 + 본문)

  python build_report.py                           # 목차 쪽 번호: LibreOffice로 렌더링해 계산
  python build_report.py --pages-from 보고서.pdf    # Word에서 PDF로 저장한 파일로 계산 (Word와 정확히 일치)

1) md에서 '본문 시작' 표시 아래만 pandoc으로 docx 변환 (## → 제목 1, ### → 제목 2)
   A4·여백 2cm인 reference 문서를 함께 넘겨, 그림 너비(width=100%)가 본문 폭에 맞게 계산되게 함
2) docx XML 후처리: 표지, 목차(Word TOC 필드), 머리글·쪽 번호, 맑은 고딕, 제목·표·캡션·상자·코드 서식,
   Word 호환성 모드 해제
3) PDF로 렌더링해 각 제목의 쪽 번호를 찾아 목차에 채움 (Word에서 목차 우클릭 → '필드 업데이트'로도 다시 계산 가능)
   - 기본: LibreOffice. 맑은 고딕·Consolas는 Windows 글꼴 폴더(WSL 포함)나 Mac용 Word 앱 안에서 찾아
     임시 프로필에 넣음. 그래도 Word와 줄바꿈이 조금 달라 문서 뒤쪽 제목이 1쪽 어긋날 수 있음
   - --pages-from: 빌드한 docx를 Word에서 PDF로 저장한 뒤 그 PDF를 넘기면 Word 기준 쪽 번호로 다시 빌드
     (목차 줄 수는 그대로라 쪽 나눔은 바뀌지 않음)

필요: pandoc, pypandoc  (쪽 번호 계산: LibreOffice, 그리고 pdftotext 또는 PyMuPDF)
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from xml.sax.saxutils import escape, unescape

import pypandoc

HERE   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(HERE, "202258096_임준현_딥러닝응용_과제2보고서.md")
OUT    = SRC.replace(".md", ".docx")
MARKER = "<!-- 본문 시작"

COVER = {
    "course":   "딥러닝응용",
    "title":    "과제 2 보고서",
    "subtitle": ["CNN 기반 MNIST 숫자 분류기", "파라미터 3만 개·연산량 0.01 GMACs 제한에서 정확도 최대화"],
    "result":   "테스트 정확도 99.64%   ·   파라미터 29,098개   ·   MACs 4,716,874",
    "info":     [("학번", "202258096"), ("이름", "임준현"), ("제출일", "2026년 11월 5일")],
}
HEADER = ("딥러닝응용 과제 2 보고서", "202258096 임준현")   # 본문 쪽 머리글 (왼쪽, 오른쪽)

#%% 디자인 상수
FONT, MONO = "Malgun Gothic", "Consolas"
NAVY, BLUE, INK, MUTED = "1F3864", "2F5496", "1A1A1A", "595959"
RULE, HEAD_FILL, BOX_FILL = "B4C3DC", "E7ECF4", "EFF3F9"
CODE_FILL, CODE_LINE = "F6F8FA", "D5DBE3"

PAGE_W, PAGE_H, MARGIN = 11906, 16838, 1134            # A4, 여백 2cm (twip)
TEXT_W = PAGE_W - 2 * MARGIN
SZ_BODY, SZ_TABLE, SZ_TABLE_CODE, SZ_CODE = 20, 18, 16, 16   # half-point (20 = 10pt)
LINE_BODY = 264                                         # 줄 간격 (240 = 1줄)

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
R_NS = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'

#%% XML 조각
def rfonts(font=FONT, east=FONT):
    return f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{east}" w:cs="{font}"/>'

def spacing(before=0, after=0, line=240):
    return f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'

def run(text, bold=False, size=None, color=None):
    rpr = ""
    if bold:  rpr += "<w:b/><w:bCs/>"
    if color: rpr += f'<w:color w:val="{color}"/>'
    if size:  rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

def para(content, style=None, jc=None, before=None, after=None, extra_ppr=""):
    ppr = f'<w:pStyle w:val="{style}"/>' if style else ""
    ppr += extra_ppr
    if before is not None or after is not None:
        ppr += f'<w:spacing w:before="{before or 0}" w:after="{after or 0}"/>'
    if jc: ppr += f'<w:jc w:val="{jc}"/>'
    return f"<w:p><w:pPr>{ppr}</w:pPr>{content}</w:p>"

def cover_xml():
    c = COVER
    rule = f'<w:pBdr><w:bottom w:val="single" w:sz="12" w:space="14" w:color="{NAVY}"/></w:pBdr>'
    xml  = para(run(c["course"], size=30, color=MUTED), jc="center", before=3400, after=200)
    xml += para(run(c["title"], bold=True, size=64, color=NAVY), jc="center", after=420, extra_ppr=rule)
    xml += para(run(c["subtitle"][0], bold=True, size=30, color=INK), jc="center", before=420, after=100)
    xml += para(run(c["subtitle"][1], size=24, color=INK), jc="center", after=0)
    xml += para(run(c["result"], size=21, color=BLUE), jc="center", before=560)
    # 학번/이름/제출일: 테두리 없는 2열 표로 정렬
    rows = ""
    indent = '<w:ind w:left="360"/>'
    for k, v in c["info"]:
        rows += ("<w:tr>"
                 f'<w:tc><w:tcPr><w:tcW w:w="1500" w:type="dxa"/></w:tcPr>{para(run(k, bold=True, size=24), jc="distribute", after=120)}</w:tc>'
                 f'<w:tc><w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>{edit_ppr(para(run(v, size=24), after=120), ind=indent)}</w:tc>'
                 "</w:tr>")
    xml += para("", before=3900)
    # tblLook 0000: Word가 기본 표 스타일의 '첫 행' 서식(아래 테두리)을 표지 표에 적용하지 않도록 함
    xml += ('<w:tbl><w:tblPr><w:tblW w:w="4500" w:type="dxa"/><w:jc w:val="center"/>'
            '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/>'
            '<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
            '<w:tblLayout w:type="fixed"/><w:tblLook w:val="0000"/></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="1500"/><w:gridCol w:w="3000"/></w:tblGrid>' + rows + "</w:tbl>")
    return xml   # 다음 쪽(목차)은 목차 제목의 '쪽 나눔 앞' 속성으로 넘김

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
    heading = para(run("목차"), style="TOCHeading", extra_ppr="<w:pageBreakBefore/>")
    # 빈 쪽 나눔 문단 대신 본문 첫 제목에 '쪽 나눔 앞'을 걸어, 목차가 쪽을 꽉 채워도 빈 쪽이 생기지 않게 함
    return ('<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents"/>'
            f'<w:docPartUnique/></w:docPartObj></w:sdtPr><w:sdtContent>{heading}{body}'
            "</w:sdtContent></w:sdt>")

HEADER_XML = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr {W_NS} {R_NS}>'
              f'<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="4" w:space="4" w:color="{RULE}"/></w:pBdr>'
              f'<w:tabs><w:tab w:val="right" w:pos="{TEXT_W}"/></w:tabs>{spacing()}</w:pPr>'
              f'{run(HEADER[0], size=16, color=MUTED)}<w:r><w:tab/></w:r>{run(HEADER[1], size=16, color=MUTED)}'
              '</w:p></w:hdr>')

FOOTER_XML = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {W_NS} {R_NS}>'
              f'<w:p><w:pPr>{spacing()}<w:jc w:val="center"/></w:pPr>'
              f'<w:r><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
              f'<w:r><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="18"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
              f'<w:r><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
              f'<w:r><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="18"/></w:rPr><w:t>1</w:t></w:r>'
              f'<w:r><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')

SECT_PR = ('<w:sectPr><w:headerReference w:type="default" r:id="rIdHeader1"/>'
           '<w:footerReference w:type="default" r:id="rIdFooter1"/>'
           f'<w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
           f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" '
           'w:header="567" w:footer="567" w:gutter="0"/><w:titlePg/></w:sectPr>')   # 표지에는 머리글·쪽 번호 없음

#%% 스타일 (styles.xml)
def style(sid, name, ppr="", rpr="", kind="paragraph", based="Normal", nxt=None, link=None, custom=True):
    return (f'<w:style w:type="{kind}"' + (' w:customStyle="1"' if custom else "") + f' w:styleId="{sid}">'
            f'<w:name w:val="{name}"/><w:basedOn w:val="{based}"/>'
            + (f'<w:next w:val="{nxt}"/>' if nxt else "") + (f'<w:link w:val="{link}"/>' if link else "")
            + "<w:qFormat/>" + (f"<w:pPr>{ppr}</w:pPr>" if ppr else "") + (f"<w:rPr>{rpr}</w:rPr>" if rpr else "")
            + "</w:style>")

H_RULE = f'<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="{RULE}"/></w:pBdr>'
STYLES = [
    style("BodyText", "Body Text", spacing(0, 120, LINE_BODY), based="Normal", link="BodyTextChar", custom=False),
    style("Compact", "Compact", spacing(0, 50, LINE_BODY), based="BodyText"),
    style("Heading1", "heading 1",
          "<w:keepNext/><w:keepLines/>" + H_RULE + spacing(440, 180) + '<w:outlineLvl w:val="0"/>',
          rfonts() + f'<w:b/><w:bCs/><w:color w:val="{NAVY}"/><w:sz w:val="30"/><w:szCs w:val="30"/>',
          nxt="BodyText", link="Heading1Char", custom=False),
    style("Heading2", "heading 2",
          "<w:keepNext/><w:keepLines/>" + spacing(300, 110) + '<w:outlineLvl w:val="1"/>',
          rfonts() + f'<w:b/><w:bCs/><w:color w:val="{BLUE}"/><w:sz w:val="23"/><w:szCs w:val="23"/>',
          nxt="BodyText", link="Heading2Char", custom=False),
    style("BoxText", "Box Text", spacing(0, 50, LINE_BODY), based="BodyText"),
    style("SourceCode", "Source Code",
          '<w:pBdr>' + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="5" w:color="{CODE_LINE}"/>'
                               for s in ("top", "left", "bottom", "right")) + "</w:pBdr>"
          f'<w:shd w:val="clear" w:color="auto" w:fill="{CODE_FILL}"/>' + spacing(100, 180)
          + '<w:ind w:left="110" w:right="110"/>',
          link="VerbatimChar"),
    style("VerbatimChar", "Verbatim Char", rpr=rfonts(MONO, FONT) + '<w:sz w:val="18"/><w:szCs w:val="18"/>',
          kind="character", based="BodyTextChar"),
    style("TableCaption", "Table Caption", "<w:keepNext/>" + spacing(220, 80),
          f'<w:color w:val="{INK}"/><w:sz w:val="18"/><w:szCs w:val="18"/>'),
    style("ImageCaption", "Image Caption", spacing(60, 240, 252),
          f'<w:color w:val="{MUTED}"/><w:sz w:val="17"/><w:szCs w:val="17"/>'),
    style("TOCHeading", "TOC Heading", H_RULE + spacing(0, 300),
          rfonts() + f'<w:b/><w:bCs/><w:color w:val="{NAVY}"/><w:sz w:val="30"/><w:szCs w:val="30"/>', custom=False),
    style("TOC1", "toc 1", spacing(110, 10), '<w:b/><w:bCs/><w:sz w:val="20"/><w:szCs w:val="20"/>',
          nxt="Normal", custom=False),
    style("TOC2", "toc 2", spacing(0, 10) + '<w:ind w:left="420"/>', '<w:sz w:val="19"/><w:szCs w:val="19"/>',
          nxt="Normal", custom=False),
]

def patch_styles(st):
    # 기본 글꼴·크기: 테마 글꼴(Aptos 등) 대신 맑은 고딕을 직접 지정, 본문 10pt, 줄 간격, 외톨이 줄 방지
    # 한글과 영문·숫자 사이 자동 간격은 끔 ("29,098개"가 "29,098 개"로 벌어지지 않게)
    st = re.sub(r"<w:rFonts [^>]*Theme[^>]*/>", rfonts(), st)
    st = re.sub(r"(<w:rPrDefault>.*?<w:sz w:val=\")\d+(\" ?/>\s*<w:szCs w:val=\")\d+",
                rf"\g<1>{SZ_BODY}\g<2>{SZ_BODY}", st, count=1, flags=re.S)
    st = re.sub(r"<w:pPrDefault>.*?</w:pPrDefault>",
                '<w:pPrDefault><w:pPr><w:widowControl/><w:autoSpaceDE w:val="0"/><w:autoSpaceDN w:val="0"/>'
                f"{spacing(0, 120, LINE_BODY)}</w:pPr></w:pPrDefault>",
                st, count=1, flags=re.S)
    st = st.replace('w:eastAsia="zh-CN"', 'w:eastAsia="ko-KR"')
    for new in STYLES:
        sid = re.search(r'w:styleId="([^"]+)"', new).group(1)
        pat = re.compile(r'<w:style [^>]*w:styleId="%s"[^>]*>.*?</w:style>' % sid, re.S)
        st = pat.sub(lambda m: new, st, count=1) if pat.search(st) else st.replace("</w:styles>", new + "</w:styles>")
    return st

def patch_numbering(num):
    """목록 들여쓰기를 줄임 (기본 0.5인치 단위 → 0.7cm 단위)"""
    def lvl(m):
        k = int(m.group(1))
        return re.sub(r'<w:ind w:left="\d+" w:hanging="\d+" ?/>',
                      f'<w:ind w:left="{400 * (k + 1)}" w:hanging="280"/>', m.group(0))
    return re.sub(r'<w:lvl w:ilvl="(\d+)".*?</w:lvl>', lvl, num, flags=re.S)

def patch_settings(s):
    """Word '호환성 모드' 해제 (Word 2013 이후 형식으로 표시)"""
    compat = ('<w:compat><w:compatSetting w:name="compatibilityMode" '
              'w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>')
    if "compatibilityMode" in s:
        return s
    for anchor in ("<w:rsids", "<w:themeFontLang", "<w:clrSchemeMapping", "<w:decimalSymbol", "</w:settings>"):
        if anchor in s:
            return s.replace(anchor, compat + anchor, 1)
    return s

#%% 본문(document.xml) 서식
PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid",
             "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
             "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"]
RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
             "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w",
             "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign",
             "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"]
CHILD_RE = re.compile(r"<w:(\w+)\b[^>]*?/>|<w:(\w+)\b[^>]*>.*?</w:\2>", re.S)
P_RE     = re.compile(r"<w:p>.*?</w:p>", re.S)
R_RE     = re.compile(r"<w:r>.*?</w:r>", re.S)
CAP_RE   = re.compile(r"^(표|그림) \d+\.")
KEEP     = "<w:keepNext/>"

def _merge(inner, tag, order, props):
    """inner의 자식 요소에 props({이름: xml 또는 None})를 덮어쓰고 스키마 순서대로 다시 조립"""
    kids = [(m.group(1) or m.group(2), m.group(0)) for m in CHILD_RE.finditer(inner)]
    kids = [(k, x) for k, x in kids if k not in props] + [(k, x) for k, x in props.items() if x]
    kids.sort(key=lambda kx: order.index(kx[0]) if kx[0] in order else len(order))
    return f"<w:{tag}>" + "".join(x for _, x in kids) + f"</w:{tag}>"

def edit_ppr(p, **props):
    m = re.match(r"<w:p>(?:<w:pPr>(.*?)</w:pPr>|<w:pPr\s*/>)?", p, re.S)
    return "<w:p>" + _merge(m.group(1) or "", "pPr", PPR_ORDER, props) + p[m.end():]

def edit_rpr(r, **props):
    m = re.match(r"<w:r>(?:<w:rPr>(.*?)</w:rPr>|<w:rPr\s*/>)?", r, re.S)
    return "<w:r>" + _merge(m.group(1) or "", "rPr", RPR_ORDER, props) + r[m.end():]

def ptext(p):
    return unescape("".join(re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>", p)))

def pstyle_of(p):
    m = re.match(r'<w:p><w:pPr><w:pStyle w:val="([^"]+)"', p)
    return m.group(1) if m else None

def size_props(sz):
    return {"sz": f'<w:sz w:val="{sz}"/>', "szCs": f'<w:szCs w:val="{sz}"/>'}

def runs_sized(p, sz_text, sz_code, bold=False):
    """문단 안 모든 run에 글자 크기를 직접 지정 (코드 run은 따로)"""
    def fix(m):
        r = m.group(0)
        code = 'w:rStyle w:val="VerbatimChar"' in r or re.search(r'w:rStyle w:val="\w+Tok"', r)
        props = size_props(sz_code if code else sz_text)
        if bold:
            props.update(b="<w:b/>", bCs="<w:bCs/>")
        return edit_rpr(r, **props)
    return R_RE.sub(fix, p)

KEEP_ROWS = 10   # 이 행 수 이하의 표는 한 쪽에 묶고, 더 긴 표는 머리글 행을 반복하며 다음 쪽으로 넘어갈 수 있음

def format_table(t):
    """본문 표: 위·아래 굵은 선, 머리글 행 음영, 가로 선만, 행 나뉨 방지.
    짧은 표는 캡션과 함께 한 쪽에 유지하고, 긴 표는 앞 3행만 캡션에 붙여 빈 공간을 줄임"""
    tblpr = (f'<w:tblPr><w:tblStyle w:val="Table"/><w:tblW w:w="5000" w:type="pct"/>'
             f'<w:tblBorders><w:top w:val="single" w:sz="10" w:space="0" w:color="{NAVY}"/>'
             '<w:left w:val="nil"/>'
             f'<w:bottom w:val="single" w:sz="10" w:space="0" w:color="{NAVY}"/>'
             '<w:right w:val="nil"/>'
             f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{RULE}"/>'
             '<w:insideV w:val="nil"/></w:tblBorders>'
             '<w:tblLayout w:type="fixed"/>'
             '<w:tblCellMar><w:top w:w="36" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
             '<w:bottom w:w="36" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar>'
             '<w:tblLook w:val="0620" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" '
             'w:noHBand="1" w:noVBand="1"/></w:tblPr>')
    t = re.sub(r"<w:tblPr>.*?</w:tblPr>", lambda m: tblpr, t, count=1, flags=re.S)
    rows = list(re.finditer(r"<w:tr>.*?</w:tr>", t, re.S))
    keep_upto = len(rows) - 1 if len(rows) <= KEEP_ROWS else 3   # 이 행 앞까지 다음 행과 붙임
    out, pos = [], 0
    for i, m in enumerate(rows):
        r = m.group(0)
        header = "<w:tblHeader" in r
        last = i >= keep_upto
        r = re.sub(r"<w:trPr>.*?</w:trPr>|<w:trPr\s*/>", "", r, flags=re.S)
        r = r.replace("<w:tr>", "<w:tr><w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if header else "") + "</w:trPr>", 1)
        tcpr = ((f'<w:tcBorders><w:bottom w:val="single" w:sz="8" w:space="0" w:color="{NAVY}"/></w:tcBorders>'
                 f'<w:shd w:val="clear" w:color="auto" w:fill="{HEAD_FILL}"/>') if header else "") + '<w:vAlign w:val="center"/>'
        r = re.sub(r"<w:tcPr\s*/>|<w:tcPr></w:tcPr>", f"<w:tcPr>{tcpr}</w:tcPr>", r)

        def cell_para(pm):
            p = edit_ppr(pm.group(0), keepNext=None if last else KEEP, spacing=spacing(0, 0, 245))
            return runs_sized(p, SZ_TABLE, SZ_TABLE_CODE, bold=header)
        r = P_RE.sub(cell_para, r)
        out.append(t[pos:m.start()] + r)
        pos = m.end()
    out.append(t[pos:])
    return "".join(out)

def box_table(paras):
    """인용 블록(>) 문단들 → 왼쪽 강조선이 있는 옅은 상자(1칸 표)"""
    inner = ""
    for j, p in enumerate(paras):
        p = edit_ppr(p, pStyle='<w:pStyle w:val="BoxText"/>',
                     keepNext=KEEP if j < len(paras) - 1 else None,
                     spacing=spacing(0, 50 if j < len(paras) - 1 else 0, LINE_BODY))
        inner += p
    return ('<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/>'
            f'<w:tblBorders><w:top w:val="nil"/><w:left w:val="single" w:sz="24" w:space="0" w:color="{BLUE}"/>'
            '<w:bottom w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
            f'<w:shd w:val="clear" w:color="auto" w:fill="{BOX_FILL}"/><w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:top w:w="120" w:type="dxa"/><w:left w:w="220" w:type="dxa"/>'
            '<w:bottom w:w="120" w:type="dxa"/><w:right w:w="220" w:type="dxa"/></w:tblCellMar>'
            '<w:tblLook w:val="0000"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{TEXT_W}"/></w:tblGrid>'
            '<w:tr><w:trPr><w:cantSplit/></w:trPr>'
            f'<w:tc><w:tcPr><w:tcW w:w="{TEXT_W}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{BOX_FILL}"/></w:tcPr>'
            f"{inner}</w:tc></w:tr></w:tbl>")

def format_flow(chunk, after_table=False):
    """표 밖의 문단들: 캡션, 그림, 코드 블록, 상자, 다음 문단과 붙여 둘 문단.
    after_table: 이 구간이 본문 표 바로 뒤에서 시작하면 첫 문단에 위 여백을 줌 (표 아래 선에 붙지 않게)"""
    ps = P_RE.findall(chunk)
    gaps = P_RE.split(chunk)                     # 문단 사이의 나머지 (공백·책갈피), len(ps) + 1개
    styles = [pstyle_of(p) for p in ps]
    out = list(ps)
    for i, p in enumerate(ps):
        st, text = styles[i], ptext(p)
        nxt = ps[i + 1] if i + 1 < len(ps) else ""
        if "<w:drawing>" in p:                                         # 그림: 가운데, 캡션과 같은 쪽
            out[i] = edit_ppr(p, keepNext=KEEP, spacing=spacing(160, 40), jc='<w:jc w:val="center"/>')
        elif CAP_RE.match(text):                                       # 표 N. / 그림 N. 캡션
            kind = "TableCaption" if text.startswith("표") else "ImageCaption"
            p = edit_ppr(p, pStyle=f'<w:pStyle w:val="{kind}"/>')
            out[i] = R_RE.sub(lambda m: edit_rpr(m.group(0), color=f'<w:color w:val="{NAVY}"/>'), p, count=1)
        elif st == "SourceCode":                                       # 코드 블록: 글자 크기
            out[i] = runs_sized(p, SZ_CODE, SZ_CODE)
        elif st in ("BodyText", "FirstParagraph", "Compact"):
            props = {}
            runs = [r for r in R_RE.findall(p) if "<w:t" in r]
            bold_only = st != "Compact" and runs and len(text) < 30 and all("<w:b/>" in r or "<w:b />" in r for r in runs)
            before_code = pstyle_of(nxt) == "SourceCode"
            before_list = st != "Compact" and "<w:numPr>" in nxt[:400] and len(text) < 80
            if bold_only or before_code or before_list:                # 소제목·코드·목록 앞 문단은 다음과 붙임
                props["keepNext"] = KEEP
            if bold_only:
                props["spacing"] = spacing(180, 60, LINE_BODY)
            elif i == 0 and after_table:
                props["spacing"] = spacing(120, 50 if st == "Compact" else 120, LINE_BODY)
            if props:
                out[i] = edit_ppr(p, **props)
    # 연속된 BlockText 문단 → 상자 하나, 상자 다음 문단은 위 여백
    res, i = gaps[0], 0
    while i < len(ps):
        if styles[i] == "BlockText":
            j = i
            while j + 1 < len(ps) and styles[j + 1] == "BlockText":
                j += 1
            res += box_table(out[i:j + 1]) + gaps[j + 1]
            if j + 1 < len(ps):
                out[j + 1] = edit_ppr(out[j + 1], spacing=spacing(200, 120, LINE_BODY))
            i = j + 1
            continue
        res += out[i] + gaps[i + 1]
        i += 1
    return res

def format_body(doc):
    i0 = doc.index("<w:body>") + len("<w:body>")
    i1 = doc.rindex("<w:sectPr")
    head, body, tail = doc[:i0], doc[i0:i1], doc[i1:]
    parts, pos = [], 0
    for m in re.finditer(r"<w:tbl>.*?</w:tbl>", body, re.S):
        parts.append(format_flow(body[pos:m.start()], after_table=pos > 0))
        parts.append(format_table(m.group(0)))
        pos = m.end()
    parts.append(format_flow(body[pos:], after_table=pos > 0))
    tail = re.sub(r"<w:sectPr\b.*?</w:sectPr>|<w:sectPr\s*/>", SECT_PR, tail, count=1, flags=re.S)
    return head + "".join(parts) + tail

HEAD_RE = re.compile(r'(?:<w:bookmarkStart w:id="\d+" w:name="([^"]+)"\s*/>\s*)?'
                     r'<w:p><w:pPr><w:pStyle w:val="Heading([12])"\s*/>.*?</w:p>', re.S)

def headings(doc):
    """(level, bookmark, text) 목록. 책갈피가 제목 문단 앞(이전 pandoc)이든 안(pandoc 3.x)이든 처리"""
    out = []
    for m in HEAD_RE.finditer(doc):
        bm = m.group(1)
        if not bm:
            inner = re.search(r'<w:bookmarkStart w:id="\d+" w:name="([^"]+)"', m.group(0))
            bm = inner.group(1) if inner else None
        if bm:
            out.append((int(m.group(2)), bm, ptext(m.group(0))))
    return out

def postprocess(src_docx, dst_docx, pages):
    tmp = dst_docx + ".tmp"
    with zipfile.ZipFile(src_docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        entries = None
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                doc = format_body(data.decode())
                entries = headings(doc)
                body_start = doc.index("<w:body>") + len("<w:body>")
                # 본문 첫 문단(1장 제목)은 새 쪽에서 시작
                first = P_RE.search(doc, body_start)
                doc = (doc[:first.start()] + edit_ppr(first.group(0), pageBreakBefore="<w:pageBreakBefore/>",
                                                      spacing=spacing(0, 180)) + doc[first.end():])
                doc = doc[:body_start] + cover_xml() + toc_xml(entries, pages) + doc[body_start:]
                data = doc.encode()
            elif item.filename == "word/styles.xml":
                data = patch_styles(data.decode()).encode()
            elif item.filename == "word/numbering.xml":
                data = patch_numbering(data.decode()).encode()
            elif item.filename == "word/settings.xml":
                data = patch_settings(data.decode()).encode()
            elif item.filename == "word/_rels/document.xml.rels":
                data = data.decode().replace(
                    "</Relationships>",
                    '<Relationship Id="rIdHeader1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                    'relationships/header" Target="header1.xml"/>'
                    '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                    'relationships/footer" Target="footer1.xml"/></Relationships>').encode()
            elif item.filename == "[Content_Types].xml":
                data = data.decode().replace(
                    "</Types>",
                    '<Override PartName="/word/header1.xml" ContentType="application/'
                    'vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>'
                    '<Override PartName="/word/footer1.xml" ContentType="application/'
                    'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>').encode()
            zout.writestr(item, data)
        zout.writestr("word/header1.xml", HEADER_XML)
        zout.writestr("word/footer1.xml", FOOTER_XML)
    os.replace(tmp, dst_docx)
    return entries

#%% pandoc 변환 (A4 reference 문서 사용)
def reference_docx(path):
    """pandoc 기본 reference.docx에 A4·여백 2cm만 지정 → width=100% 그림이 본문 폭(17cm)에 맞음"""
    pandoc = pypandoc.get_pandoc_path()
    raw = subprocess.run([pandoc, "--print-default-data-file", "reference.docx"],
                         capture_output=True, check=True).stdout
    src = path + ".src"
    with open(src, "wb") as f:
        f.write(raw)
    sect = (f'<w:sectPr><w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/><w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" '
            f'w:bottom="{MARGIN}" w:left="{MARGIN}" w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>')
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                d = data.decode()
                d2 = re.sub(r"<w:sectPr\b.*?</w:sectPr>|<w:sectPr\s*/>", sect, d, count=1, flags=re.S)
                data = (d2 if d2 != d else d.replace("</w:body>", sect + "</w:body>")).encode()
            zout.writestr(item, data)
    os.remove(src)

#%% 쪽 번호 계산 (LibreOffice 렌더링)
FONT_FILES = ("malgun.ttf", "malgunbd.ttf",                                # 맑은 고딕 (본문)
              "consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf")  # Consolas (코드)

def font_files():
    """Word와 같은 글꼴 파일 찾기: Windows(WSL 포함), Mac용 Word 앱 안, 사용자 글꼴 폴더 (파일 이름 대소문자 무시)"""
    dirs = ["/mnt/c/Windows/Fonts", "C:/Windows/Fonts",
            "/Applications/Microsoft Word.app/Contents/Resources/DFonts",
            "/Library/Fonts", os.path.expanduser("~/Library/Fonts"), os.path.expanduser("~/.local/share/fonts")]
    found = {}
    for d in dirs:
        names = {n.lower(): n for n in os.listdir(d)} if os.path.isdir(d) else {}
        for f in FONT_FILES:
            if f not in found and f in names:
                found[f] = os.path.join(d, names[f])
    return found

def render_pdf(docx, td):
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    work = os.path.join(td, "render")
    prof = os.path.join(work, "lo")
    fontdir = os.path.join(prof, "user", "fonts")      # LibreOffice 사용자 글꼴 폴더 (임시 프로필)
    os.makedirs(fontdir, exist_ok=True)
    fonts = font_files()
    for f in fonts.values():
        shutil.copy(f, fontdir)
    if "malgun.ttf" not in fonts:
        print("맑은 고딕 파일을 찾지 못함: 목차 쪽 번호가 Word와 다를 수 있음")
    env = dict(os.environ)
    if sys.platform.startswith("linux") and fonts:     # WSL/Linux: fontconfig에도 등록
        conf = os.path.join(work, "fonts.conf")
        with open(conf, "w") as fp:
            fp.write('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
                     '<include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
                     f'<dir>{fontdir}</dir><cachedir>{work}/cache</cachedir></fontconfig>')
        env["FONTCONFIG_FILE"] = conf
    # 렌더링용 사본: ASCII 파일 이름, run마다 붙은 한국어 언어 태그 제거
    # (태그가 있으면 LibreOffice가 영문·숫자 중간에서 줄을 바꿔 Word와 쪽 나눔이 달라짐. 제출 파일은 그대로)
    src = os.path.join(work, "report.docx")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(src, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = re.sub(rb'<w:lang w:val="ko-KR"\s*/>', b"", data)
            zout.writestr(item, data)
    try:
        subprocess.run([soffice, "--headless", f"-env:UserInstallation={pathlib.Path(prof).as_uri()}",
                        "--convert-to", "pdf", "--outdir", work, src],
                       env=env, capture_output=True, timeout=300, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print("LibreOffice 렌더링 실패:", e)
        return None
    pdf = os.path.join(work, "report.pdf")
    return pdf if os.path.exists(pdf) else None

def pdf_page_texts(pdf):
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", pdf, "-"], capture_output=True)
        if r.returncode == 0:
            return r.stdout.decode("utf-8", errors="replace").split("\f")
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            return None
    return [p.get_text() for p in pymupdf.open(pdf)]

def find_pages(docx, entries, td, pages_from=None):
    """pages_from: Word에서 내보낸 PDF 등 이미 렌더링된 PDF (없으면 LibreOffice로 렌더링)"""
    pdf = pages_from or render_pdf(docx, td)
    texts = pdf_page_texts(pdf) if pdf else None
    if not texts or len(texts) < 3:
        print("PDF를 만들거나 읽지 못함 (LibreOffice·pdftotext/PyMuPDF 확인): "
              "목차 쪽 번호는 Word에서 '필드 업데이트'로 채워야 함")
        return {}
    norm = lambda s: re.sub(r"\s+", "", unicodedata.normalize("NFC", s))
    pages, start = {}, 2   # 표지(1), 목차(2) 이후부터 검색
    for _, bm, t in entries:
        key = norm(t)
        for p in range(start, len(texts)):
            lines = [norm(l) for l in texts[p].splitlines()]
            # 제목이 한 줄이거나, 두 줄로 나뉜 경우까지 비교
            if key in lines or key in (a + b for a, b in zip(lines, lines[1:])):
                pages[bm] = p + 1
                start = p
                break
        else:
            print("쪽 번호를 찾지 못함:", t)
    print(f"렌더링 {len(texts) - (texts[-1].strip() == '')}쪽")
    return pages

#%%
def main():
    ap = argparse.ArgumentParser(description="보고서 md → docx (표지·목차 포함)")
    ap.add_argument("--pages-from", metavar="PDF",
                    help="목차 쪽 번호를 이 PDF에서 찾음 (예: 한 번 빌드한 docx를 Word에서 PDF로 저장한 파일). "
                         "지정하지 않으면 LibreOffice로 렌더링해 계산")
    args = ap.parse_args()
    if args.pages_from and not os.path.isfile(args.pages_from):
        sys.exit(f"--pages-from 파일이 없음: {args.pages_from}")
    with open(SRC, encoding="utf-8") as f:
        md = f.read()
    body = md[md.index("\n", md.index(MARKER)) + 1:]
    with tempfile.TemporaryDirectory() as td:
        body_md, ref, raw_docx, pass1 = (os.path.join(td, n) for n in
                                         ("body.md", "reference.docx", "raw.docx", "pass1.docx"))
        with open(body_md, "w", encoding="utf-8") as f:
            f.write(body)
        reference_docx(ref)
        pypandoc.convert_file(body_md, "docx", outputfile=raw_docx,
                              extra_args=[f"--resource-path={HERE}", f"--reference-doc={ref}",
                                          "--from=markdown-implicit_figures+tex_math_dollars",
                                          "--shift-heading-level-by=-1", "--metadata=lang:ko-KR"])
        # 1차: 쪽 번호 없이 임시 파일로 생성 → 렌더링해서 쪽 번호 계산 → 2차: 쪽 번호 채워서 OUT에 저장
        entries = postprocess(raw_docx, pass1, pages={})
        pages = find_pages(pass1, entries, td, pages_from=args.pages_from)
        try:
            postprocess(raw_docx, OUT, pages)
        except PermissionError:
            sys.exit(f"{OUT}에 쓸 수 없음: Word에서 보고서 docx를 닫고 다시 실행")
    print(f"saved {OUT}")
    for lvl, bm, t in entries:
        print(f"{'  ' * (lvl - 1)}{t} ... {pages.get(bm, '?')}")

if __name__ == "__main__":
    main()
