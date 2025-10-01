# 전처리 변경 로그 및 세부사항

이 문서는 데이터 전처리 **버전별(v0, v1, v2)** 적용 내용을 정리한 문서입니다.  
각 버전에서 어떤 처리를 했는지 명확히 기록합니다.

---

## 버전별 상세 비교표

| 항목 | v0 | v1 | v2 | 비고 |
|---|---|---|---|---|
| 데이터 분할 | train/test 80/20 (random_state=42) | 동일 |  | 동일 |
| 범주형 캐스팅 | `mold_code`, `EMS_operation_time` → object | 동일 |  | 동일 |
| date/time swap & 타입 변환 | swap 후 `date` -> `%Y-%m-%d`, `time` -> `%H:%M:%S` | 동일 |  | 동일 |
| 파생변수 | `hour`, `weekday` 추가 | 동일 |  | 동일 |
| 불필요 컬럼 제거 | `id`, `line`, `name`, `mold_name`, `registration_time` 삭제 | 동일 |  | 동일 |
| 범주형 처리를 위한 컬럼 제거 | `date`, `time` 삭제 | 동일 |  | 동일 |
| `emergency_stop` 결측 처리 | 결측 행 제거 | 동일 |  | 동일 (긴급중단 행 제거) |
| `upper_mold_temp3`, `lower_mold_temp3` 등 센서 컬럼 | 삭제 (`upper_mold_temp3`, `lower_mold_temp3`, `heating_furnace`, `molten_volume`) | 동일 |  | 센서 이상/결측으로 제거 |
| `molten_temp` 결측 처리 | 전체 평균(mean) 대체 (`SimpleImputer`) | 그룹(`mold_code`) 선형 보간 (테스트: ffill으로 보간) |  | v1은 그룹 단위 보간으로 정보 보존 강화 |
| 중복 / 겹치는 행 처리 | 없음 | 동일 `mold_code`에서 연속으로 같은 `count` 값인 행 제거 (8722,8412,8573,8917,8600 대상) |  | v1에 추가된 핵심 처리 |
| 이상치 처리 (개별) | 특정 제거 없음 | `upper_mold_temp2`의 특정 이상치(예: idx 42632) 삭제 |  | v1에서 추가 |
| `tryshot_signal` 처리 | 결측 → 'N' (학습에 그대로 둠; 테스트 시 D는 불량 처리) | 결측 → 'N' `tryshot_signal == 'D'` 행 학습에서 제거 |  | v1은 D 행을 학습에서 제외 |