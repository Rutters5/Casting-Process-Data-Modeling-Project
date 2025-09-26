# modules/tab_process_explanation.py
from shiny import ui

def panel():
    return ui.nav_panel(
        "공정 설명",
        ui.layout_column_wrap(
            ui.card(
                ui.h3("공정 개요"),
                ui.markdown(
                    """
**목표**  
- 금형/주물/사출 공정에서의 불량 발생 메커니즘을 이해하고, 주요 공정변수와의 인과적 연결을 설명합니다.

**데이터 기반 접근**  
- 수집 변수: 압력/속도/온도/두께 등 연속 변수
- 레이블: 양품/불량 (이진 분류)
- 특징: 생산 로트·금형별 분포 차이, 계절/설비조건에 따른 드리프트 가능성
                    """
                ),
            ),
            ui.card(
                ui.h3("주요 공정 변수와 영향"),
                ui.markdown(
                    """
1. **cast_pressure (주입압력)**  
   - 캐비티 충진 안정성에 직접 영향 → 과소압력 시 쇼트(충진 불량), 과대압력 시 버(Burr)·치수편차 위험.

2. **sleeve_temperature (슬리브 온도)**  
   - 용탕 점도 및 유동성 결정 → 너무 낮으면 미충진/냉결, 너무 높으면 가스 결함·기포 증가.

3. **upper_mold_temp1/2, lower_mold_temp1/2 (금형 온도)**  
   - 응고 속도·수축 균일성에 영향 → 온도 편차가 크면 내부 응력/크랙·기공 발생 가능.

4. **low_section_speed / high_section_speed (사출 속도 구간)**  
   - 충진 패턴 제어 → 초기 저속 안정 주입, 고속 구간에서 웰드라인/기포 최소화가 핵심.

5. **biscuit_thickness (비스킷 두께)**  
   - 러너/게이트 유동의 간접 지표 → 과소 두께는 유동 제한, 과대는 과충진·사이클 타임 손실.
                    """
                ),
            ),
            ui.card(
                ui.h3("불량 유형과 원인 매핑(예시)"),
                ui.markdown(
                    """
- **미충진/쇼트**: 낮은 *cast_pressure*, 낮은 *sleeve_temperature*, 금형 온도 불균일  
- **기공/가스결함**: 과도한 *high_section_speed*, 과고온 *sleeve_temperature*, 배기 불량  
- **치수 불량/변형**: 금형 온도 편차, 급랭·잔류응력, 과압  
- **표면 결함(웰드라인/흐름자국)**: 속도 전환 타이밍 불량, 온도·압력 세팅 미스
                    """
                ),
            ),
            ui.card(
                ui.h3("운영 팁(현장 적용 체크리스트)"),
                ui.markdown(
                    """
- 파라미터 변경 시 **단일 변수 원칙**으로 영향분석(DoE 권장)  
- 금형 온도 **상/하 균형** 유지(편차 최소화)  
- 속도 구간 전환 포인트와 압력 상승 타이밍 **로그 기반 검증**  
- **주기적 스케일링 재적용** 및 설비별 **보정값 관리**
                    """
                ),
            ),
            fill=False,  # 카드 폭이 너무 넓어지지 않게
        ),
    )

def server(input, output, session):
    # 이 탭은 출력/반응 없음 (텍스트 전용)
    pass
