from shiny import App, render, ui
from modules import (
    tab_analysis_copy_copy as tab_analysis_copy,
    tab_model_performance,
    tab_process_explanation,
    tab_preprocessing,
)

TAB_DEFINITIONS = [
    {
        "id": "analysis",
        "label": "예측 분석",
        "icon": "fa-solid fa-chart-line",
        "content": tab_analysis_copy.panel_body(),
    },
    {
        "id": "performance",
        "label": "모델 성능 평가",
        "icon": "fa-solid fa-gauge-high",
        "content": tab_model_performance.panel_body(),
    },
    {
        "id": "eda",
        "label": "EDA 분석",
        "icon": "fa-solid fa-chart-pie",
        "content": tab_process_explanation.panel_body(),
    },
    {
        "id": "preprocess",
        "label": "데이터 전처리 요약",
        "icon": "fa-solid fa-list-check",
        "content": tab_preprocessing.panel_body(),
    },
]

TAB_CONTENT = {tab["id"]: tab["content"] for tab in TAB_DEFINITIONS}
DEFAULT_TAB = TAB_DEFINITIONS[0]["id"]

app_assets = """
<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\">
<style>
body {
    background: #383636;
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
}

.outer-container {
    background: #000000;
    border-radius: 32px;
    padding: 16px;
    margin: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    min-height: calc(100vh - 40px);
}

.inner-container {
    border-radius: 24px;
    overflow: hidden;
    min-height: calc(100vh - 72px);
    position: relative;
}

.sidebar-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 24px;
}

.sidebar-toggle-button {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    background: rgba(255, 255, 255, 0.08);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    z-index: 10;
}

.sidebar-toggle-button:hover {
    background: rgba(74, 144, 226, 0.85);
    border-color: rgba(74, 144, 226, 0.9);
}

.sidebar-toggle-button i {
    font-size: 18px;
    transition: transform 0.3s ease;
}

.sidebar-toggle-button.collapsed i {
    transform: rotate(180deg);
}

.bslib-sidebar-layout {
    transition: grid-template-columns 0.3s ease, margin 0.3s ease;
}

.bslib-sidebar-layout > aside {
    transition: transform 0.3s ease, opacity 0.3s ease, padding 0.3s ease;
}

body.sidebar-collapsed .bslib-sidebar-layout {
    grid-template-columns: 68px 1fr !important;
}

body.sidebar-collapsed .bslib-sidebar-layout > aside {
    transform: translateX(calc(-100% + 68px));
    padding: 24px 12px !important;
}

body.sidebar-collapsed .sidebar-title,
body.sidebar-collapsed #sidebar-nav {
    opacity: 0;
    pointer-events: none;
}

body.sidebar-collapsed .sidebar-header {
    justify-content: flex-end;
}

body.sidebar-collapsed .sidebar-toggle-button {
    background: rgba(74, 144, 226, 0.9);
    border-color: rgba(74, 144, 226, 1);
}

body.sidebar-collapsed .bslib-sidebar-layout > div.main {
    padding-left: 24px !important;
}


.dashboard-page {
    height: 100%;
}

.bslib-sidebar-layout {
    height: 100%;
    background: transparent !important;
}

.bslib-sidebar-layout > aside {
    background: #2A2D30 !important;
    border: none !important;
    padding: 32px 20px !important;
    display: flex !important;
    flex-direction: column;
    gap: 24px;
}

.sidebar-shell {
    width: 100%;
}

.sidebar-title {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    text-align: left;
}


.sidebar-title span:last-child {
    font-size: 12px;
    opacity: 0.7;
    letter-spacing: 0.2em;
}

#sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 12px;
    color: #ecf0f1;
    font-weight: 600;
    font-size: 15px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.sidebar-nav-item:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateX(4px);
}

.sidebar-nav-item.active {
    background: #4A90E2;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.25);
}

.sidebar-nav-item i {
    width: 20px;
    text-align: center;
}

.bslib-sidebar-layout > div.main {
    background: #F3F4F5 !important;
    padding: 32px !important;
    display: flex;
    justify-content: center;
    overflow: auto;
}

.main-scroll-container {
    flex: 1 1 auto;
    max-width: 1400px;
}
</style>
<script>
(function() {
    function initSidebar() {
        const nav = document.getElementById('sidebar-nav');
        const hidden = document.getElementById('active_tab');
        if (!nav || !hidden || !window.Shiny) {
            return;
        }

        function setActive(tabId, emit) {
            if (!tabId) {
                return;
            }
            nav.querySelectorAll('.sidebar-nav-item').forEach((el) => {
                el.classList.toggle('active', el.dataset.tab === tabId);
            });
            hidden.value = tabId;
            if (emit) {
                window.Shiny.setInputValue('active_tab', tabId, { priority: 'event' });
            }
        }

        nav.querySelectorAll('.sidebar-nav-item').forEach((el) => {
            if (el.dataset.bound === 'true') {
                return;
            }
            el.dataset.bound = 'true';
            el.addEventListener('click', () => setActive(el.dataset.tab, true));
        });

        const layout = document.querySelector('.bslib-sidebar-layout');
        const toggleBtn = document.getElementById('sidebar-toggle');
        const collapsed = document.body.classList.contains('sidebar-collapsed');
        if (layout) {
            layout.classList.toggle('collapsed', collapsed);
        }
        if (toggleBtn) {
            toggleBtn.classList.toggle('collapsed', collapsed);
            if (!toggleBtn.dataset.bound) {
                toggleBtn.dataset.bound = 'true';
                toggleBtn.addEventListener('click', () => {
                    const next = !document.body.classList.contains('sidebar-collapsed');
                    document.body.classList.toggle('sidebar-collapsed', next);
                    if (layout) {
                        layout.classList.toggle('collapsed', next);
                    }
                    toggleBtn.classList.toggle('collapsed', next);
                });
            }
        }

        const initial = hidden.value;
        setActive(initial, false);

        if (window.Shiny.addCustomMessageHandler) {
            window.Shiny.addCustomMessageHandler('set-active-tab', (msg) => {
                if (msg && msg.id) {
                    setActive(msg.id, Boolean(msg.emit));
                }
            });
        }
    }

    if (document.readyState !== 'loading') {
        initSidebar();
    } else {
        document.addEventListener('DOMContentLoaded', initSidebar);
    }

    document.addEventListener('shiny:connected', initSidebar);
})();
</script>
"""


def _nav_item(tab):
    classes = ["sidebar-nav-item"]
    if tab["id"] == DEFAULT_TAB:
        classes.append("active")
    return ui.div(
        ui.tags.i(class_=tab["icon"]),
        ui.span(tab["label"]),
        class_=" ".join(classes),
        **{"data-tab": tab["id"]},
    )


SIDEBAR_NAV = ui.div(*(_nav_item(tab) for tab in TAB_DEFINITIONS), id="sidebar-nav")

sidebar = ui.sidebar(
    ui.div(ui.input_text("active_tab", None, value=DEFAULT_TAB), style="display:none;"),
    ui.div(
        ui.div(
            ui.span("불량 원인 분석"),
            ui.span("대시보드"),
            class_="sidebar-title",
        ),
        ui.tags.button(
            ui.tags.i(class_="fa-solid fa-chevron-left"),
            id="sidebar-toggle",
            class_="sidebar-toggle-button",
            type="button",
        ),
        class_="sidebar-header",
    ),
    SIDEBAR_NAV,
    class_="sidebar-shell",
    open="always",
)

app_ui = ui.page_fluid(
    ui.HTML(app_assets),
    ui.div(
        ui.div(
            ui.page_sidebar(
                sidebar,
                ui.div(
                    ui.output_ui("active_tab_content"),
                    class_="main-scroll-container",
                ),
                class_="dashboard-page",
                fillable=True,
            ),
            class_="inner-container",
        ),
        class_="outer-container",
    ),
)


def server(input, output, session):
    @render.ui
    def active_tab_content():
        tab_id = input.active_tab() or DEFAULT_TAB
        return TAB_CONTENT.get(tab_id, TAB_CONTENT[DEFAULT_TAB])

    tab_analysis_copy.server(input, output, session)
    tab_model_performance.server(input, output, session)
    tab_process_explanation.server(input, output, session)
    tab_preprocessing.server(input, output, session)


app = App(app_ui, server)







