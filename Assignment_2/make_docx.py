"""보고서 md → docx 변환. 반드시 Assignment_2/ 에서 실행 (figures/ 상대경로 임베딩)."""
import os

import pypandoc

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SRC = "202258096_임준현_딥러닝기초_과제2보고서.md"
DST = "202258096_임준현_딥러닝기초_과제2보고서.docx"

pypandoc.convert_file(SRC, "docx", outputfile=DST,
                      extra_args=["--resource-path=.", "--toc", "--toc-depth=2"])
print(f"saved: {DST}")
