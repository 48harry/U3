# MEMORY.md — 설계 결정 기록

이 파일은 **왜 그렇게 만들었는지**를 남기는 곳이다. 무엇이 어디에 있는지는 `readme.md` 5절, 코드 동작은 각 스크립트 docstring을 본다.
결정이 바뀌면 해당 항목을 고치고 맨 아래 변경 이력에 한 줄 남긴다. 코드에서 읽을 수 있는 내용은 적지 않는다.

## 1. 프로젝트 목표

- 데이터: RSW 용접기 고장 예측 벤치마크(Wang et al., Sci Data 2024, `docs/RSW Paper.pdf`). train 72파일(E01~E04 × 18), test 8파일, 1 Hz × 7일.
- 논문의 2-stage(센서 예측 → 진단 규칙)는 오류 누적으로 정확도 68%/오알람 56% → **우리는 1-step 이상탐지**로 간다 (readme 2절).
- 최종 파이프라인: 이상탐지 시그널 → 온톨로지/RAG 원인 분석(E01~E04 매뉴얼) → LLM 리포트 → Slack (readme 3절). 지금은 이상탐지 API까지 구현.

## 2. 데이터에 대해 확정된 사실 (EDA, 72파일)

- **`error` 컬럼은 이벤트가 아니라 상태 코드**다. 같은 코드가 수 시간 지속(E003 중앙값 1.5~2h, 최장 28h). 샘플 수로 세면 안 되고 에피소드(연속 구간)로 센다. gap이 에피소드를 끊지 않도록 코드를 gap 너머로 carry한다.
- **클래스별 종료 코드는 거의 결정적**: E01→E012, E02→E016, E03→E028, E04→E029 (18/18, 17/18, 18/18, 18/18), 모두 고장 전 10분 안에만 등장. 파일 마지막 행의 error는 항상 `0`이라 "마지막 행"이 아니라 "마지막 수 분"을 봐야 한다.
- E003·E011·E029는 며칠 전부터 장시간 지속되므로 단독 전조로는 약하다. readme의 "잔고장 빈도 = 전조" 가설은 부분적으로만 성립.
- 결측은 행 단위 block-out. E01/03/04는 ~1%, gap 99%가 ≤35초. **E02 8개 파일(0,2,3,4,5,6,7,8)은 64~84%** → 논문 규칙(40%)으로 폐기. E02_2는 10분짜리 데모 파일.
- c16(판 두께 setpoint) >5 또는 ≤0 = 캡 드레싱 등 비용접 작업, 파일당 1~45%.
- c7·c8·c9는 파일 내 상수(건 고유 속성). c11·c12는 누적 카운터라 절대값이 건마다 다르다. c4·c18은 거의 이진, c17 거의 상수.
- 시각은 CSV의 `...Z`를 **naive UTC**로 통일한다(`pd.to_datetime(utc=True).tz_localize(None)`). 로컬(KST) 변환 금지 — `hour_sin/cos`가 틀어진다.
- E02 일부는 2019-12~2020-07, 나머지는 2021-08~10 수집. 시기 차이가 분포 차이를 만들 수 있음(미검증).
- 논문 타깃: E02는 c2(전극 압력), 나머지는 c5(밸런스 압력). 원본 벤치마크 코드는 이와 달리 c1~c5를 타깃으로 씀.

## 3. 전처리 결정 (`.py/preprocess.py`)

| 결정 | 이유 |
|---|---|
| 결측률 >40% 파일 폐기 | 논문 규칙. 80%대 파일은 ffill해도 상수 덩어리 |
| gap ≤60초 ffill, 초과 gap은 행 삭제 + `segment_id` 분리 | gap 99%가 ≤35초. 윈도우가 구멍을 넘지 않게 |
| 비용접 행은 센서값 blank → 직전 용접값 carry(길이 제한 없음), 행은 유지 + `non_welding` 플래그 | 논문 방식. 60초 제한을 걸었더니 고장 직전 1시간이 날아가는 파일(E01_0)이 있었음 |
| c11·c12 → `welds_delta`, `pos_delta`, `welds_10min` | 건 간 비교 가능한 값으로 |
| `error_active`, `error_share_10min`, `terminal_code` 파생 | 상태 코드를 피처와 레이블 양쪽으로 쓰기 위해 |
| `ttf_s` + `label`(마지막 3600초=1) | 지도/비지도 양쪽에서 쓸 수 있게. 윈도우 길이는 옵션 |
| z-score를 **정상행(`label==0 & error_active==0`)** 으로만 fit, `scaler.json` 저장 | 비지도 학습 관례. 테스트/서빙에 동일 변환 적용 |
| `error_share_10min`, `weld_duty_10min`은 스케일링 제외 | 이미 0~1 비율. 정상행 기준 z-score가 30 같은 값을 만들었음 |
| 기본은 global 스케일, `--scale per-file`·`--drop-static` 옵션 | 건 간 오프셋 제거는 실험 대상 |

## 4. 모델링 결정 (`.py/train.py`)

- 60초 윈도우 집계(연속 피처 mean+std = 48 피처). 37M 초 단위 행을 그대로 쓰지 않는다 — 메모리와 시간 문제, 그리고 윈도우 통계가 시간 맥락을 준다.
- **분할은 파일(=건) 단위**, 클래스별 층화. 같은 건의 행이 train/val에 섞이면 안 됨.
- 정상 = `label==0 & error_active==0`. 비용접 윈도우 포함 여부는 `--exclude-non-welding`으로 실험.
- 임계값 = 학습 정상 점수 99% 분위수(=오탐률 1% 설계). `sustain=3` 연속 초과를 "지속 알람"으로 정의 — readme 3절의 "연속 N회일 때만 RAG 트리거"에 대응.
- 평가는 AUROC/AUPRC(고장 전 1h vs 정상) + 정상 알람률 + 클래스별 + 파일별 "고장으로 이어지는 마지막 알람 구간 시작 시각"과 "24h 이전 오탐 에피소드/일". "최초 알람 시각"은 초반 오탐에 걸려 무의미해서 버렸다.
- 번들(joblib dict)에 `feature_reference`(정상 윈도우 중앙값)를 넣는다 — 서빙에서 피처 기여도 계산용.
- 베이스라인(IsolationForest) 결과: 전체 AUROC 0.639, 고장 전 1h recall 5%, 정상 알람률 1.5%. **기준선일 뿐이고 종료 상태(E029 등)도 못 잡는다.**

## 5. 서빙 계약 (`main.py`)

- 입력은 CSV 컬럼 그대로(`time, c1~c19, error`). 서버가 건별 30분 링버퍼를 유지하고 전처리를 온라인으로 재현한다. 클라이언트는 아무 전처리도 하지 않는다.
- 출력 `AnomalyResult` 필드는 온톨로지 단계와의 **계약**이다: `is_anomaly, anomaly_score, threshold, score_z, severity(normal|warning|critical), sustained_alarm, consecutive_alarms, contributing_features[], context{latest_error_code, known_code_class_hint, ...}, model{}`. 필드를 빼거나 의미를 바꾸면 하류가 깨진다 — 추가는 자유.
- `severity: critical` = `sustain` 연속 초과. 이게 RAG 트리거 기준.
- `contributing_features`는 모델 무관 perturbation(피처 하나를 정상 기준값으로 치환했을 때의 점수 감소). 모델을 바꿔도 이 방식은 유지 가능.
- `known_code_class_hint`는 모델과 무관하게 2절의 종료 코드 매핑으로 채운다. 모델이 약할 때 온톨로지가 쓸 수 있는 안전장치.
- 워밍업(60초 미만)은 202 + `WarmingUp` 스키마.

## 6. 코드 규약

- 스크립트는 `.py/`, API는 루트 `main.py`(`uvicorn main:app`). `main.py`는 `sys.path`에 `.py/`를 넣어 `train`, `preprocess`를 import한다 — 번들 언피클에 `train.PCADetector`가 필요하다.
- 모든 경로는 `PROJECT_ROOT = __file__ 기준` 으로 잡는다. cwd 의존 금지 (IDE에서 `.py/`가 cwd가 되어 KeyError가 났던 적 있음). 파일을 못 찾으면 즉시 `SystemExit`로 경로를 찍는다.
- 수치는 float32, 저장은 parquet(pyarrow 없으면 csv 폴백).
- 데이터 폴더(`train/`, `preprocessed/`) 안에 산출물을 쓰지 않는다. 결과는 `eda_out/`, `models/`, `results/`로.
- 그래프 색은 클래스 고정: E01 `#2a78d6`, E02 `#eb6834`, E03 `#1baf7a`, E04 `#eda100`. 이중 축 금지.
- 환경: Windows, CPU-only torch, darts 0.41, pandas 2.3, sklearn 1.7, fastapi 0.121, pydantic 2.13. `accelerator="gpu"` 하드코딩 금지.
- 무거운 작업(72파일 EDA 40분, 벤치마크 학습 수십 시간)은 백그라운드로 여러 개 띄우지 않는다 — 메모리 부족으로 강제 종료된 적 있음. `--max-files`/`--pattern`으로 축소 테스트 먼저.

## 7. 미결 사항 / 알려진 함정

1. `models/baseline_iforest.joblib`은 `feature_reference` 추가 이전 학습본 → 재학습 전까지 API 기여도 기준값이 0.
2. E01_0은 고장 직전 1시간의 85%가 `non_welding`. 72파일에서 같은 패턴이면 `non_welding`이 레이블 누설이 되므로 입력에서 제외해야 한다.
3. `Benchmark_fixed.py`는 임포트·경로만 검증됨. 실제 학습 경로 미검증.
4. 서빙은 트레일링 60초, 학습은 분 경계 정렬 윈도우. 차이는 작지만 존재.
5. 다음 실험 후보: per-file 스케일 + static 제거, 윈도우 확대(5~10분), 시퀀스 오토인코더, `label`을 쓰는 지도학습 비교, E02의 수집 시기 분포 차이 확인.
6. 테스트 8파일(`test/`)은 아직 어느 스크립트도 손대지 않았다.

## 변경 이력

- 2026-09-22: 최초 작성. Benchmark 리뷰/수정, EDA, 전처리, IsolationForest 베이스라인, FastAPI 서빙까지의 결정 정리.
