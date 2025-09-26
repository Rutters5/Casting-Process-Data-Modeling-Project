# modules/tab_preprocessing.py
from shiny import ui, render
from pathlib import Path

# 프로젝트 루트 기준 PDF 경로(필요시 경로 조정)
# 예: <repo_root>/reports/preprocessing_report.pdf
BASE_DIR = Path(__file__).resolve().parents[2]
PDF_FILE = BASE_DIR / "reports" / "preprocessing_report.pdf"

def panel():
    return ui.nav_panel(
        "전처리 요약",
        ui.layout_column_wrap(
            ui.card(
                ui.h3("데이터 전처리 개요"),
                ui.markdown(
                    """
다음 항목들은 본 프로젝트에서 적용한 전처리 절차의 요약입니다.

- **결측치 처리**: 단일/다중 대치, 삭제 기준, 컬럼별 정책 요약  
- **이상치 처리**: 기준(통계·공정 지식), 처리 방식(윈저라이징/제거/대체)  
- **스케일링/정규화**: 표준화(예: StandardScaler), 로그 변환 등  
- **인코딩**: 범주형 원-핫/타깃 인코딩 정책  
- **특성 선택/생성**: 상관도/피처 중요도 기반 제거·생성 기준  
- **데이터 누수 방지**: 스플릿 이후 변환적용 원칙
                    """
                ),
            ),
            ui.card(
                ui.h3("자세한 전처리 보고서"),
                ui.p("아래 버튼을 눌러 전처리 상세 문서를 PDF로 다운로드하세요."),
                ui.download_button("download_pdf", "PDF 다운로드"),
                ui.div(ui.output_text("pdf_status"), class_="text-muted mt-2"),
            ),
            fill=False,
        ),
    )

def server(input, output, session):
    # PDF 존재 여부 안내
    @output
    @render.text
    def pdf_status():
        if PDF_FILE.exists():
            return f"파일 위치: {PDF_FILE}"
        else:
            return "주의: PDF 파일이 아직 존재하지 않습니다. reports/preprocessing_report.pdf 경로에 파일을 생성해 주세요."

    # PDF 다운로드 핸들러
    @render.download(filename="preprocessing_report.pdf")
    def download_pdf():
        # 존재하면 그대로 스트리밍
        if PDF_FILE.exists():
            with open(PDF_FILE, "rb") as f:
                # 큰 파일도 안전하게 전송
                chunk = f.read(8192)
                while chunk:
                    yield chunk
                    chunk = f.read(8192)
        else:
            # 파일이 없을 때 빈 PDF라도 내려주고 싶다면, 간단한 PDF 바이트를 만들어서 전달할 수도 있습니다.
            # (여기서는 매우 단순한 placeholder PDF를 생성)
            # 참고: 외부 라이브러리 없이 최소 헤더/본문으로 만든 단순 PDF (리더 compatibility는 제한적일 수 있음)
            minimal_pdf = (
                b"%PDF-1.4\n"
                b"1 0 obj<<>>endobj\n"
                b"2 0 obj<< /Type /Catalog /Pages 3 0 R >>endobj\n"
                b"3 0 obj<< /Type /Pages /Kids [4 0 R] /Count 1 >>endobj\n"
                b"4 0 obj<< /Type /Page /Parent 3 0 R /MediaBox [0 0 595 842] /Contents 5 0 R >>endobj\n"
                b"5 0 obj<< /Length 44 >>stream\n"
                b"BT /F1 24 Tf 72 770 Td (Preprocessing report missing) Tj ET\n"
                b"endstream endobj\n"
                b"6 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
                b"xref\n0 7\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000102 00000 n \n0000000158 00000 n \n0000000254 00000 n \n0000000346 00000 n \n"
                b"trailer<< /Size 7 /Root 2 0 R >>\nstartxref\n420\n%%EOF\n"
            )
            yield minimal_pdf
