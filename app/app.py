# app.py
from shiny import App, ui
from modules import (
    tab_analysis_copy,
    tab_model_performance,
    tab_process_explanation,
    tab_preprocessing,
)

# 전역 CSS 스타일
app_css = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
/* 전체 페이지 배경 */
body {
    background: #383636;
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* 외부 컨테이너 */
.outer-container {
    background: #000000;
    border-radius: 32px;
    padding: 16px;
    margin: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    min-height: calc(100vh - 40px);
}

/* 내부 컨테이너 */
.inner-container {
    background: linear-gradient(to right, #2A2D30 0%, #2A2D30 16.67%, #F3F4F5 16.67%, #F3F4F5 100%);
    border-radius: 24px;
    overflow: hidden;
    min-height: calc(100vh - 72px);
}

/* 레이아웃 */
.bslib-sidebar-layout {
    border-radius: 24px !important;
    overflow: hidden !important;
    background: transparent !important;
}

/* 사이드바 (네비게이션) 영역 */
.bslib-sidebar-layout > aside {
    background: transparent !important;
    border: none !important;
    padding: 24px 16px !important;
    margin: 0 !important;
}

/* 사이드바 상단에 제목 추가 */
.bslib-sidebar-layout > aside::before {
    content: "불량 원인 분석\\A대시보드";
    white-space: pre;
    display: block;
    color: white;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    line-height: 1.4;
}

/* 메인 콘텐츠 영역 */
.bslib-sidebar-layout > div.main {
    background: transparent !important;
    padding: 0 !important;
}

/* 네비게이션 링크 */
.nav-pills {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

.nav-pills .nav-link {
    color: #ecf0f1 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}

.nav-pills .nav-link i {
    font-size: 16px;
    width: 20px;
    text-align: center;
}

.nav-pills .nav-link:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    transform: translateX(4px);
}

.nav-pills .nav-link.active {
    background: #4A90E2 !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3) !important;
}
</style>
"""

# 앱 UI 정의 (NavPanel에 대해 인덱싱/중복래핑 X)
app_ui = ui.page_fluid(
    ui.HTML(app_css),
    ui.div(
        ui.div(
            ui.navset_pill_list(
                # 각 모듈의 panel()이 NavPanel을 반환한다고 가정하고 그대로 넣습니다.
                tab_analysis_copy.panel(),
                tab_model_performance.panel(),
                tab_process_explanation.panel(),
                tab_preprocessing.panel(),
                id="main_nav",
                widths=(2, 10),
            ),
            class_="inner-container",
        ),
        class_="outer-container",
    ),
)

# 서버 정의
def server(input, output, session):
    tab_analysis_copy.server(input, output, session)
    tab_model_performance.server(input, output, session)
    tab_process_explanation.server(input, output, session)
    tab_preprocessing.server(input, output, session)

# 앱 실행 객체
app = App(app_ui, server)
