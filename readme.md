## RSW 용접기 고장 예측 — 이상탐지 파이프라인

저항 점용접기(RSW)의 센서 시계열에서 고장 전조를 실시간으로 탐지하고, 그 신호를 원인 분석(RAG/LLM) 단계로 넘기는 프로젝트다.

- **0~4절**: 데이터와 논문, 접근 방식, 데이터에서 확인한 사실 — *무엇을 왜 하는가*
- **5절**: 구현 워크플로우 — *무엇을 어떤 순서로 돌리면 무엇이 나오는가*
- `MEMORY.md`: 설계 결정과 그 근거. 결정이 바뀌면 거기를 고친다.

표기: **〔확인〕** = 72파일 EDA로 실측 확인(5.3절), **〔수정〕** = 초기 가정이 EDA 후 바뀐 것, **〔메모〕** = 당시 판단·미결 사항.

---

### 0. 데이터 파일

https://drive.google.com/drive/folders/10u8Z5YJIqq2KRrzuE7Jn9QsCtgTgLftQ?usp=sharing

내려받아 프로젝트 루트에 `train/`(72개 CSV), `test/`(8개 CSV)로 둔다. 두 폴더는 git에 올리지 않는다(`.gitignore`). 원본 총 4,243만 행, 약 5 GB(train 4.4 GB, test 572 MB).

---

### 1. 데이터셋과 논문 리뷰

출처: Wang et al., *Scientific Data* 2024 (`docs/RSW Paper.pdf`).

#### 1.1 개요

- 자동차 차체 공장에서 실제로 3년 이상 가동된 **저항 점용접기(RSW) 80대**에서 수집한 다변량 시계열. 용접기의 제어 오류는 예기치 않은 라인 정지를 유발하므로, 과거 관측으로 **기계적 결함을 사전에 예측하고 정비 계획을 세우는 것**이 목적이다.
- 논문 원문: "The resistance spot welding (RSW) welding gun fault prediction benchmark data set has **72 multivariate time series in the training set and 8 in the testing set.** Each time series length 604800(apx.) sampled at 1 Hz with missing values and has 20 dimensions (c1-c19 and the error code). We retain the missing value and the outliers of the welding gun time series for the potential of imputation research in the future."
- "RSW guns are highly nonlinear dynamical machinery systems with interesting physical phenomena over multiple time-dependent components."

**데이터 수집 과정**

![The constructive process and the description of the welding gun fault prediction benchmark data set](image.png)

#### 1.2 파일 구성

| 항목 | 내용 |
|---|---|
| 컬럼 | `time`, `c1`~`c19`(센서 19종), `error`(상태 코드) — 총 21개 |
| 주기 | 1 Hz(1초), 파일당 약 604,800행 = 7일 |
| train | `E01_0.csv` ~ `E01_17.csv` 식으로 클래스별 18개 × 4클래스 = **72개** |
| test | `test_0.csv` ~ `test_7.csv` = **8개**. 파일명에 클래스가 없다 |
| 총량 | 이론상 80 × 604,800 ≈ 4,840만 행, 결측을 뺀 실제 **4,243만 행** |

- 각 파일은 해당 고장이 발생한 시점에서 **거꾸로 7일치**를 담는다. 즉 파일의 마지막 행이 고장 시점이다.
- **〔확인〕** E01·E03·E04는 72개 전부 정확히 7.00~7.01일이다. **E02만 예외**로 0.01일(E02_2, 10분짜리 데모 파일)부터 7일까지 제각각이다.
- **〔메모〕** 데이터 양 자체는 학습에 충분하다. 이 정도면 ML이나 1D CNN은 수월하게 돌아간다.

#### 1.3 센서 파라미터 19종

영업 비밀 보호를 위해 **정확한 명칭을 마스킹해 `c1`~`c19`로 표기**한다.

| | 이름 | | 이름 |
|---|---|---|---|
| c1 | Electrode cap offset | c11 | Welding point count |
| c2 | Electrode force | c12 | Position count |
| c3 | Electrode position | c13 | Setpoint of counterbalance pressure |
| c4 | Force build-up | c14 | Setpoint of electrode force |
| c5 | Balance pressure | c15 | Setpoint of electrode position |
| c6 | Friction | c16 | Setpoint of sheet thickness |
| c7 | Maximum aperture | c17 | Setpoint of velocity |
| c8 | Maximum electrode force | c18 | Setpoint of force build-up |
| c9 | Start friction | c19 | Offset value in robot |
| c10 | US2 (on/off) | | |

- 마스킹 때문에 도메인 해석에는 한계가 있으나, 벤치마크 코드에서 역으로 알 수 있는 것이 있다. 예를 들어 **`c16`(판 두께 setpoint)이 임계값을 넘으면 용접이 아니라 캡 교체·드레싱 같은 비용접 작업**이다. 논문 본문은 "6 초과"라 쓰지만 벤치마크 코드는 5를 쓰므로, 우리는 코드 쪽(`c16 > 5` 또는 `c16 ≤ 0`)을 따랐다(5.4절).
- **〔확인〕** `c7`·`c8`·`c9`는 한 파일 안에서 상수다(72개 중 69개). 건별 고유 속성으로 봐야 한다. `c11`·`c12`는 누적 카운터라 절대값이 건마다 다르다.

#### 1.4 `error` 컬럼은 '파일명 클래스'와 별개다

여기가 이 데이터셋에서 가장 헷갈리는 지점이다.

- **`error` 컬럼**은 1초마다 컨트롤러가 내뱉은 **내부 상태 코드**다. 전체 80파일에 등장하는 값은 `0`과 아래 11종이 전부다.

  `E001, E003, E006, E007, E009, E010, E011, E012, E016, E028, E029`

  대부분은 기계를 멈출 정도가 아닌 잔고장·일시적 통신 불량·정비 알림이다.
- **파일명의 E01~E04**는 그 7일의 끝에서 실제로 라인을 세운 **중대 정지 사유**다. 코드 체계가 아예 다르다(`E01` ≠ `E001`).
- **〔수정〕** 초기에는 "`error` 컬럼에 잔고장이 평소보다 자주 뜨는 것 자체가 전조"라고 봤고, 이를 입력 피처로 쓰기로 했다. EDA 결과 **절반만 맞다**: `error`는 이벤트가 아니라 **몇 시간씩 지속되는 상태**이며(E003 에피소드 중앙값 1.5~2시간, 최장 28시간), E003·E011·E029는 며칠 전부터 계속 떠 있어 단독 전조로는 약하다. 그래서 빈도 대신 `error_share_10min`(최근 10분 중 에러 상태였던 비율)로 만들어 넣었다(5.3~5.4절).
- **〔확인〕** 반대로 **클래스별 '종료 코드'는 거의 결정적**이다. 고장 10분 안에 E01→E012(18/18), E02→E016(17/18), E03→E028(18/18), E04→E029(18/18)가 나타난다. 다만 **파일 마지막 행의 코드는 항상 `0`**이라 "마지막 행"이 아니라 "마지막 수 분"을 봐야 한다. 이 매핑은 두 곳에 쓴다 — 테스트 8파일의 클래스 추론(5.5절), API의 `known_code_class_hint`(5.7절).

#### 1.5 4가지 고장 클래스

| 코드 | 상태 명칭 | 발생 조건 (논문 번역) | 종료 코드 |
|---|---|---|---|
| **E01** | Counterbalance timeout (카운터밸런스 시간 초과) | 지정된 보상 압력에 1800 ms 동안 도달하지 못함 | E012 |
| **E02** | Electrode broke (전극 파손) | 전극 위치가 기준 이동의 영점보다 작음 | E016 |
| **E03** | Unwanted movement (원치 않는 움직임) | 실제 위치 값이 실린더 스트로크의 6.5%를 초과해 이탈 | E028 |
| **E04** | Drift (표류/이탈) | 잠긴 실린더가 정지하지 않고 분당 5 mm 이상 움직임 | E029 |
| E00 | Normal state (정상) | 에러 없이 정상 작동 | — |

**E00(정상) 파일은 없다.** 4가지 고장 케이스에 대해서만 18개씩 존재한다. 정상 구간은 각 파일의 고장 이전 부분에서 얻는다 — 이것이 지도학습 대신 **정상 데이터만으로 학습하는 이상탐지**를 택한 이유 중 하나다(2절).

#### 1.6 결측치·이상치를 그대로 남긴 이유

논문은 센서 통신망 패킷 손실로 인한 **결측치와 비용접 동작으로 생긴 이상치를 의도적으로 제거하지 않았다**. 불완전한 실제 산업 데이터를 다루는 강건한 모델 개발을 장려하기 위해서다.

**〔메모 → 결론〕** 처음엔 "정확도를 위해 보간할지, robustness를 위해 남길지"가 미결이었다. 결국 **규칙을 정해 처리하는 쪽**으로 갔다: 짧은 gap은 채우고, 긴 gap은 세그먼트를 끊고, 비용접 구간은 플래그를 달아 남긴다(5.4절). 이유는 4절과 `MEMORY.md` 3절에 있다.

---

### 2. 접근 방식 — 논문의 2-stage 대신 1-step 이상탐지

#### 2.1 논문이 한 것

고장 여부를 바로 분류하지 않고 **2단계**로 간다.

1. **Stage 1**: 타깃 파라미터(E02는 전극 압력 `c2`, 나머지는 밸런스 압력 `c5`)의 미래 수치를 시계열로 예측
2. **Stage 2**: 예측된 수치에 기존 고장 진단 규칙(1.5절의 발생 조건)을 적용해 고장 여부를 판정

10개 모델 비교 결과 예측 정확도는 **TFT(Temporal Fusion Transformers)**와 **Random Forest**가 가장 우수했고, 단순 RNN도 안정적이었다.

그런데 그 예측값으로 실제 고장을 판정해 보면:

| 지표 | 값 |
|---|---|
| 평균 정확도 | 68.62% |
| 정상을 고장으로 오인(위양성) | 56% |

#### 2.2 다르게 가는 이유

- **오류가 누적된다.** 미래 60초 센서값을 먼저 예측하고(Stage 1) 그 위에 규칙을 얹으면(Stage 2), 1단계 예측 오차가 2단계 판정 오차로 증폭된다. 위양성 56%가 그 결과다.
- **결측·이상치를 그대로 둔 것**도 모델 오차에 기여했다.
- **〔메모〕** 2-stage 구조를 제대로 재현·튜닝하는 것은 현실적으로 부담이 컸다. 우리 목표는 논문 재현이 아니라 쓸 만한 경보 신호를 만드는 것이다.

#### 2.3 채택한 구조

> 센서를 예측하고 → 그 위에 규칙을 얹어 에러를 판정 **(2-stage)**
> 　　　⇩
> 센서 흐름에서 **곧바로 이상 여부를 판정 (1-step)**

구체적으로는 **정상 구간만으로 학습하는 비지도 이상탐지**다. 고장 케이스가 클래스당 18건뿐이라 지도학습을 하기엔 양성 표본이 너무 적고, 정상 구간은 파일마다 6일 이상 확보되기 때문이다. 세부 설계는 5.6절, 근거는 `MEMORY.md` 4절.

**〔수정〕** 초기 구상은 "진단 규칙으로 도출된 최종 에러(E01~E04)를 정답(Target)으로 삼아 센서 패턴에서 역산하는 분류(Classification)"였다. 이건 **채택하지 않았다.** 대신 정상 대비 이탈 정도를 점수로 내고, 클래스 식별은 1.4절의 종료 코드 매핑(`known_code_class_hint`)으로 보조한다. 모델이 약해도 하류 온톨로지가 클래스를 알 수 있게 하는 안전장치다.

**〔메모〕** 작업 비중은 대략 전처리 50%, 학습·튜닝 30%, 하류 연결 20% 정도로 예상했고, 실제로도 전처리가 가장 오래 걸렸다.

---

### 3. 하류 파이프라인 구상 — 원인 분석과 알림

이상탐지 신호가 나온 다음 단계다. **현재 1~2번까지의 입력 계약(`AnomalyResult`)만 구현되어 있고, 3번 이후는 미구현이다**(5.7절, 5.10절).

1. 필터링된 강한 이상 시그널이 탐지되면 Root Cause Agent로 전달 → **구현됨**: `severity: critical`(3회 연속 임계 초과)이 트리거 기준
2. Agent가 이상 감지 시점 앞뒤의 센서 수치·로그 스트림을 모아 컨텍스트 구성 → **부분 구현**: `contributing_features`(어떤 센서가 점수를 올렸는지)와 `context`(에러 코드, 용접 활동량)를 API가 이미 내보낸다
3. 감지된 에러 코드에 맞는 정비 지침서를 Vector DB에서 RAG로 검색 → **미구현**
4. LLM이 이상 패턴과 검색된 매뉴얼을 융합해 원인·영향도·조치 가이드를 자연어로 요약 → **미구현**

이후 계획:

- 작업자에게 Slack으로 심각도·이상 항목·요약 원인을 실시간 전송
- 필요시 Markdown/PDF 1-page 장애 분석 리포트 자동 생성

설계상 유의점:

- **알람 피로도.** 너무 예민하면 무시당한다. 그래서 임계값을 오탐률 1% 설계로 잡고, **연속 3회 초과일 때만** RAG를 트리거하도록 했다(5.6~5.7절).
- **매뉴얼 RAG 구성.** E01~E04는 발생 조건이 명확하므로(1.5절) 클래스별 조치 매뉴얼을 넣어두면 원인 분석 리포트가 성립한다. **〔메모〕** 다만 매뉴얼이 어떤 형태로 주어지는지, 특정 에러에 대한 명확한 해결책이 있는지는 아직 확인 못 했다.

---

### 4. 데이터에서 확인한 사실

EDA(5.3절) 전의 관찰과, 72파일 실측으로 확정된 내용이다. 여기서 나온 수치가 전처리 규칙(5.4절)의 근거가 된다.

#### 4.1 결측률의 극단적 편차

- E01·E03·E04는 파일당 결측률 **0.7~2.6%**로 양호하다. gap의 99%가 35초 이하다.
- **E02만 심각하다.** 18개 중 **8개가 64~84%**로, 수집 기간 대부분 센서 통신이 끊겨 있다.

| 파일 | 결측률 | | 파일 | 결측률 |
|---|---|---|---|---|
| E02_0 | 81.4% | | E02_5 | 82.8% |
| E02_2 | 82.7% | | E02_6 | 84.5% |
| E02_3 | 64.4% | | E02_7 | 71.2% |
| E02_4 | 82.1% | | E02_8 | 82.6% |

→ 논문 규칙(결측률 40% 초과 폐기)을 적용해 이 8개를 버렸다. **64/72 파일**이 남는다. 살아남은 E02도 최대 39.8%라 클래스 중 가장 나쁘다.

**〔확인〕** 결측은 컬럼 단위가 아니라 **행 단위 block-out**이다. 특정 센서만 비는 게 아니라 그 초의 데이터 전체가 없다. 그래서 gap 60초까지는 직전 값으로 채우고, 그보다 긴 gap은 행을 지우고 **세그먼트를 분리**한다 — 윈도우가 구멍을 넘어가면 안 되기 때문이다.

#### 4.2 수집 시기의 차이

E01·E03·E04는 2021년 8~10월에 몰려 있는데, **E02는 2019년 12월~2020년 7월 데이터가 다수 섞여 있다.** 시기 차이가 분포 차이를 만들 수 있다(미검증, `MEMORY.md` 7절 미결 사항).

#### 4.3 비용접 구간 (c16 이상치)

`c16`(판 두께 setpoint)이 5 초과이거나 0 이하면 용접이 아닌 작업(캡 드레싱·교체)이다. **〔확인〕** 파일당 **0.2~44.6%, 평균 11.3%**를 차지한다. 지워버리면 고장 직전 구간이 통째로 날아가는 파일이 있어서, 센서값은 직전 용접값으로 이어 붙이고 행은 남기되 `non_welding` 플래그를 단다(5.4절).

#### 4.4 타깃의 이진화

> **초기 구상:** 20개가 넘는 에러 코드를 전부 예측하면 오알람이 폭증한다. 파일 마지막에 집중 기록되는 코드만 '치명적 고장(Target=1)'으로 정의하고 나머지는 정상(0)으로 병합해 이진 분류로 단순화하자.

**〔수정〕** 두 군데가 바뀌었다.

- 코드는 20개가 아니라 **11종**이다(1.4절).
- 레이블을 **종료 코드가 아니라 시간 기준**으로 잡았다. 종료 코드는 고장 10분 전에야 뜨기 때문에 그걸 정답으로 쓰면 "고장 10분 전 탐지"밖에 안 된다. 대신 `ttf_s`(고장까지 남은 초)를 계산해 **마지막 3600초를 `label=1`**로 둔다. 종료 코드는 별도 플래그(`terminal_code`)와 클래스 힌트로만 쓴다.

#### 4.5 잔고장 로그를 입력 피처로

> **초기 구상:** 잔고장(E016 등)이 반복해서 튀는 현상 자체가 큰 고장의 전조일 수 있다. `error`를 정답지로만 쓰지 말고 "최근 10분 누적 횟수" 같은 파생 변수로 만들어 입력에 넣자.

**〔수정〕** 방향은 맞지만 **횟수가 아니라 비율**이어야 한다. `error`는 이벤트가 아니라 지속 상태라서(1.4절) 횟수로 세면 28시간짜리 상태가 1회로 잡힌다. 그래서 `error_share_10min`(최근 10분 중 에러 상태 비율)로 넣었다. 전조로서의 가치는 제한적이라는 것도 확인됐다(1.4절).

#### 4.6 기타

- **E02_2는 결측이 아니라 10분짜리 데모 파일이다.** 원본 데이터가 없어 데모로 대체된 것으로 보이며, 7일치가 아니라 0.01일치다. 결측률 82.7%로 어차피 폐기된다.
- **〔확인〕** 센서 간 상관: `c2`↔`c4` +0.94, `c2`↔`c18` +0.89, `c2`↔`c13` −0.88 (논문 SCM 재현, `eda_out/scm_overall.csv`).

---

### 5. 구현 워크플로우 (2026-09-23 기준)

설계 결정의 **이유**는 `MEMORY.md`에, 각 단계의 **동작**은 스크립트 docstring에 있다. 이 절은 "무엇을 어떤 순서로 돌리면 무엇이 나오는지"만 적는다.

#### 5.0 한눈에 보기

```
                 ┌──────────────────────────────────────────────────────────────────┐
 train/*.csv ──► │ .py/eda.py                                                       │──► eda_out/
  (72파일)        │   탐색 · 전처리 규칙의 근거                                          │
                 └──────────────────────────────────────────────────────────────────┘
 train/*.csv ──► .py/preprocess.py ─────────► preprocessed/*.parquet
                                                + scaler.json, preprocess_config.json, manifest.csv
                                                        │ (설정·스케일러 재사용)
 test/*.csv ───► .py/preprocess.py --split test ───────┴► preprocessed/test/*.parquet + manifest.csv
  (8파일)
 preprocessed/ ─► .py/train.py ─────────────► models/baseline_iforest.joblib   (모델 + 스케일러 + 전처리 설정 + 검증/테스트 지표)
                                              models/baseline_iforest_metrics.json
                  .py/train.py --evaluate ──► models/baseline_iforest_test_metrics.json, models/scores/test_N_iforest.csv
 models/ ───────► main.py (FastAPI) ────────► AnomalyResult JSON ──► 온톨로지 / RAG / LLM 리포트 / Slack (3절, 미구현)
```

| 단계 | 명령 | 입력 → 출력 | 소요 |
|---|---|---|---|
| 0. 스모크 테스트 | `pytest` | 합성 데이터 → 임시 폴더 | 15초 |
| 1. EDA (선택) | `python .py/eda.py` | `train/` → `eda_out/` | 40분 |
| 2. 전처리(학습) | `python .py/preprocess.py` | `train/` → `preprocessed/` | 7분 |
| 3. 전처리(테스트) | `python .py/preprocess.py --split test` | `test/` + 2단계 산출물 → `preprocessed/test/` | 40초 |
| 4. 학습·평가 | `python .py/train.py` | `preprocessed/` → `models/` | 1분 |
| 4'. 테스트 재평가 | `python .py/train.py --evaluate` | `models/` + `preprocessed/test/` → `models/` | 20초 |
| 5. 서빙 | `uvicorn main:app --reload` | `models/` + 센서 스트림 → JSON | — |

한 번에: `python .py/preprocess.py && python .py/preprocess.py --split test && python .py/train.py` (약 9분). 순서가 중요하다 — 3단계는 2단계의 `scaler.json`·`preprocess_config.json`을 읽고, 4단계는 3단계 산출물이 있어야 테스트 지표를 낸다.

#### 5.1 환경과 파일 배치

```
U3/
├─ train/  test/           원본 CSV (0절 링크, git 제외)
├─ preprocessed/           2·3단계 산출물 (git 제외)
│   └─ test/
├─ eda_out/                1단계 산출물
├─ models/                 4단계 산출물 (joblib 번들, 지표 JSON, scores/)
├─ .py/                    스크립트: eda.py, preprocess.py, train.py, Benchmark.py(원본), Benchmark_fixed.py,
│                          error_check.py·file_check.py(원본 CSV 점검용 단발 스크립트)
├─ main.py                 FastAPI 서빙
├─ tests/test_pipeline.py  E2E 스모크 테스트
├─ requirements.txt  pyproject.toml  (ruff/black 120자, Benchmark.py·pic code/ 제외)
├─ MEMORY.md               설계 결정 기록 · CLAUDE.md  작업 규약 · docs/  논문
└─ pic code/               논문 그림 재현 스크립트 (파이프라인과 무관)
```

- Python 3.11, CPU 전용. `pip install -r requirements.txt`.
- 모든 스크립트는 `__file__` 기준으로 프로젝트 루트를 잡으므로 어느 디렉터리에서 실행해도 된다. 데이터 폴더(`train/`, `test/`, `preprocessed/`) 안에는 산출물을 쓰지 않는다.
- 무거운 작업(EDA, 벤치마크)은 동시에 여러 개 띄우지 않는다(메모리 부족으로 강제 종료된 적 있음). `--pattern "E04_*" --max-files 3`으로 먼저 축소 실행.

#### 5.2 (참고) 벤치마크 코드 리뷰 — `Benchmark.py` → `Benchmark_fixed.py`

논문의 Stage 1(시계열 예측) 원본은 그대로 실행되지 않는다. 고친 것:

- 모든 플롯이 `predl_NBEATS`(마지막에야 정의됨)를 그림 → `NameError`
- 테스트 시계열 전체를 넣고 그 **끝 이후**를 예측해 실측과 겹치는 구간이 없음 → 마지막 `N_PRED`초를 홀드아웃
- PNG 파일명이 `file_name[-4:]`(=".csv")라 테스트 파일끼리 덮어씀
- `accelerator="gpu"` 하드코딩(이 PC는 CPU torch) → `"auto"`
- TFT는 future covariate가 없으면 `add_relative_index=True` 필요
- `darts.mape`는 실측에 0이 있으면 예외(c2·c4는 50~85%가 0) → NaN 처리, MARRE 추가
- pandas 2.x에서 제거된 `resample('S')`, `fillna(method=)`, `to_numeric(errors='ignore')`
- 선행 결측 bfill, 이상치 마스킹을 원-핫 **이전**에 수행, 죽은 `switch` 로직 제거
- 논문 규칙대로 결측률 40% 초과 파일 폐기

10개 모델 전체 학습은 수십 시간 규모라 축소 스모크 테스트만 시도했고, 메모리 부족으로 중단되어 **학습~평가 경로는 실측 검증되지 않았다**(임포트·API 명·경로만 확인). 우리 파이프라인(1-step 이상탐지, 2절)은 이 코드를 쓰지 않는다.

#### 5.3 1단계 — EDA (`python .py/eda.py`)

산출물: `eda_out/eda_report.md`(요약), `file_summary.csv`, `error_episodes.csv`, `scm_*.csv`, `01~09_*.png`, `files/<파일>.png`

핵심 발견 → 전처리·모델링 결정:

| 발견 | 수치 | 결정 |
|---|---|---|
| 결측은 행 단위 block-out | E01/03/04 0.7~2.6%, gap 99%가 ≤35초, 최장 ~1h. **E02 8개 파일이 64~84%** | 40% 초과 폐기(논문), ≤60초 ffill, 긴 gap은 세그먼트 분리 |
| c16 이상치 = 캡 드레싱 | 파일당 0.2~44.6%, 평균 11.3% | 센서값 carry-forward + `non_welding` 플래그 |
| **error 컬럼은 이벤트가 아니라 상태** | E003 에피소드 중앙값 1.5~2h, 최장 28h | 에피소드 단위 집계, `error_active`/`error_share_10min` 피처 |
| 클래스별 종료 코드가 거의 결정적 | E01→E012 18/18, E02→E016 17/18, E03→E028 18/18, E04→E029 18/18 (모두 고장 전 10분 내) | `terminal_code` 플래그, 테스트 파일 클래스 추론, API의 `known_code_class_hint` |
| c7·c8·c9는 파일 내 상수 | 69/72 파일 | 건 고유 속성(정적 공변량), `--drop-static` 옵션 |
| c11·c12는 누적 카운터 | 절대값이 건마다 다름 | 증분 + 10분 롤링 합으로 대체 |
| 수집 시기 | E02 일부는 2019-12~2020-07, 나머지는 2021-08~10 | 타임라인 확인용 `04_collection_timeline.png` |
| 타깃 상관 | c2↔c4 +0.94, c2↔c18 +0.89, c2↔c13 −0.88 | 논문 SCM 재현(`scm_overall.csv`) |

2절의 "잔고장 코드를 전조 피처로" 가설은 절반만 맞다: 종료 코드는 고장 10분 전에야 나타나고, E003·E011·E029는 며칠 전부터 장시간 지속되는 상태라 단독 전조로는 약하다.

#### 5.4 2단계 — 전처리, 학습 파일 (`python .py/preprocess.py`)

처리 순서(파일마다): 1 Hz 격자 정렬 → 결측률 >40% 파일 폐기 → error 코드 gap 너머로 carry → gap ≤60초 ffill, 초과 gap은 행 삭제·`segment_id` 분리(`fill_gaps()`, 서빙과 공유) → c16 규칙으로 비용접 행 센서값 blank + carry, `non_welding` 플래그 → c11/c12 → 증분·10분 롤링 → 시각 피처, `ttf_s`, `label` → 정상행(`label==0 & error_active==0`)으로 z-score fit(글로벌) → 저장.

출력 `preprocessed/`:

| 파일 | 내용 |
|---|---|
| `<파일>.parquet` | 1 Hz, float32. `file, class, gun, segment_id` · 센서 `c1~c9, c13~c19`(z-score), `c10`(0/1) · 파생 `welds_delta, pos_delta, welds_10min, weld_duty_10min, error_share_10min, hour_sin, hour_cos, dow` · 상태 `error_code, error_active, terminal_code, non_welding` · 레이블 `ttf_s`, `label`(마지막 3600초=1) |
| `manifest.csv` | 파일별 행·세그먼트·결측률·비용접 비율·폐기 사유·`class_source` |
| `scaler.json` | 스케일 컬럼별 mean/std (3단계·서빙이 읽음) |
| `preprocess_config.json` | 사용한 옵션 (3단계·4단계 번들이 읽음) |

주요 옵션: `--max-missing-rate 0.4`, `--gap-fill-limit 60`, `--pre-failure-window 3600`, `--resample 10s`, `--scale global|per-file|none`, `--drop-static`. `per-file`·`--resample`은 실험용이다 — 서빙이 재현할 수 없어 그 번들은 API가 기동을 거부한다.

결과(2026-09-23): 64/72 파일, 37.5M행, 1,047 세그먼트. 폐기 8개는 모두 E02(결측 64~84%).

#### 5.5 3단계 — 전처리, 테스트 파일 (`python .py/preprocess.py --split test`)

같은 코드 경로를 타되 세 가지가 다르다.

- 명시하지 않은 옵션은 2단계의 `preprocess_config.json`에서 가져온다. z-score는 `scaler.json`을 적용만 하고 **재적합하지 않는다**.
- 파일명(`test_N`)에 클래스가 없으므로 마지막 10분의 종료 코드로 추론해 `class`에 넣고, 근거를 `manifest.csv`의 `class_source`에 남긴다(여러 코드가 있으면 고장에 가장 가까운 것: test_3은 E016→E029라 E04).
- 출력은 `preprocessed/test/`. 2단계 산출물을 덮어쓰지 않고 `scaler.json`도 쓰지 않는다.

결과: 8/8 유지, 4.83M행. 추론 클래스 E01×2(test_0,1), E02×1(test_2), E03×2(test_4,5), E04×3(test_3,6,7). test_2·test_3은 행의 87~95%가 `non_welding`이라 학습 최대치(45%)를 크게 넘는다(5.10절 참고).

#### 5.6 4단계 — 학습·평가 (`python .py/train.py`)

- 세그먼트 안에서 60초 윈도우 집계(연속 피처 24개 × mean/std = 48 피처, 학습 ~60만 윈도우)
- **파일(=건) 단위** 클래스별 층화 분할, 25% 검증(14개 건)
- IsolationForest(300 trees), 정상 윈도우(`label==0 & error_active==0`) 464,395개로 학습
- 임계값 = 학습 정상 점수의 99% 분위수(0.5411, 오탐률 1% 설계). `sustain=3` 연속 초과 = 지속 알람
- 평가 = AUROC/AUPRC(고장 전 1h vs 정상), 정상 알람률, 클래스별, 파일별 "고장으로 이어지는 마지막 알람 구간 시작 시각"과 "24h 이전 오탐 에피소드/일"
- 검증 후 `preprocessed/test/`가 있으면 **같은 임계값으로 테스트도 자동 평가**해 `metrics["test"]`에 넣는다(`--no-test`로 생략)

검증(14개 건):

| | AUROC | recall@1h | 정상 알람률 |
|---|---|---|---|
| 전체 | 0.630 | 0.044 | 0.014 |
| E01 | 0.675 | 0.016 | 0.008 |
| E02 | 0.821 | 0.278 | 0.043 |
| E03 | 0.630 | 0.000 | 0.018 |
| E04 | 0.575 | 0.008 | 0.003 |

테스트(8개 건, 클래스는 추론값):

| | AUROC | recall@1h | 정상 알람률 |
|---|---|---|---|
| 전체 | 0.687 | 0.054 | 0.032 |
| E01 (2) | 0.452 | 0.000 | 0.002 |
| E02 (1) | 0.800 | 0.167 | 0.110 |
| E03 (2) | 0.887 | 0.017 | 0.000 |
| E04 (3) | 0.678 | 0.077 | 0.044 |

→ 오탐률은 설계대로 1%대지만 **고장 전 1시간을 거의 못 잡는다**. 종료 상태(E029 등)도 임계값 아래로 본다. 기준선일 뿐이다.

산출물 `models/`:

| 파일 | 내용 |
|---|---|
| `baseline_iforest.joblib` | `{model, feature_cols, model_cols, window, threshold, sustain, scaler, preprocess_config, feature_reference, feature_scale, metrics, train_files, val_files, args, created}` — 서빙에 필요한 것이 전부 들어 있다 |
| `baseline_iforest_metrics.json` | 위에서 `model`만 뺀 것 |
| `baseline_iforest_test_metrics.json` | `--evaluate` 결과 |
| `scores/test_N_iforest.csv` | `--evaluate`가 쓰는 파일별 윈도우 점수(`score, alarm, ttf_s, label, error_active, non_welding`) |

다른 진입점: `--evaluate`(학습 없이 저장 번들로 테스트만), `--score <parquet>`(파일 하나 스코어링 → `models/scores/`), `--model pca`, `--window 120`, `--exclude-non-welding`, `--max-files 8`(빠른 확인). 코드에서는 `from train import load_bundle, score_frame`.

#### 5.7 5단계 — 실시간 추론 API (`uvicorn main:app --reload`)

`main.py`는 `sys.path`에 `.py/`를 넣어 `train`·`preprocess`를 import하고, `RSW_MODEL_PATH`(기본 `models/baseline_iforest.joblib`)의 번들을 기동 시 로드한다. 번들에 든 `scaler`와 `preprocess_config`(gap 한도, c16 임계)로 전처리를 온라인으로 재현하므로 클라이언트는 아무 전처리도 하지 않는다.

| 메서드 | 경로 | 역할 |
|---|---|---|
| POST | `/predict` | `{gun_id, readings:[SensorReading]}` → 건별 30분 버퍼에 추가, 최신 60초 윈도우 스코어링. 60초 미만이면 `202 warming_up` |
| GET | `/model` | 모델 카드: 윈도우·임계값·피처·전처리 파라미터·검증/테스트 지표·파일 목록 |
| GET | `/health`, `/guns` | 상태 / 건별 버퍼·연속 알람 |
| DELETE | `/guns/{gun_id}` | 버퍼 리셋(정비 후) |

입력 `SensorReading` = CSV 컬럼 그대로(`time, c1~c19, error`; `c10`은 on/off·bool 허용, `error`는 `^(0|E\d{3})$`).

출력 `AnomalyResult` — 온톨로지 단계와의 **계약**(필드 삭제·의미 변경 금지, 추가는 자유):

```json
{"gun_id": "G0", "status": "ok", "window_start": "...", "window_end": "...", "n_samples": 60, "history_s": 1809,
 "is_anomaly": true, "anomaly_score": 0.61, "threshold": 0.54, "score_z": 4.6,
 "severity": "critical", "sustained_alarm": true, "consecutive_alarms": 3,
 "contributing_features": [{"feature": "c5_mean", "sensor": "c5", "sensor_name": "Balance pressure",
                            "statistic": "mean", "value": -3.1, "reference": 0.02, "contribution": 0.08, "share": 0.41}],
 "context": {"latest_error_code": "E029", "error_active_share": 1.0, "non_welding_share": 0.0,
             "welds_in_window": 0, "welds_10min": 12, "weld_duty_10min": 0.02, "known_code_class_hint": "E04"},
 "model": {"name": "baseline_iforest", "type": "iforest", "created": "...", "window_s": 60, "threshold": 0.54,
           "threshold_q": 0.99, "sustain": 3, "n_features": 48}}
```

- `severity`: `normal`(임계 미만) / `warning`(초과) / `critical`(`sustain`=3회 연속 초과) — 3절의 "연속 N회일 때만 RAG 트리거" 기준
- `contributing_features`: 피처 하나를 정상 기준값(`feature_reference`)으로 치환했을 때의 점수 감소량, 상위 8개. 모델 무관 방식
- `known_code_class_hint`: 최신 error 코드가 종료 코드면 그 클래스. 모델과 무관한 안전장치
- 요청당 ~40 ms. 실데이터 검증: test_3 마지막 30분을 300행씩 보내면 마지막 윈도우의 API 점수가 오프라인 parquet에서 계산한 점수와 일치(0.5808)
- `per-file` 스케일·`--resample` 번들은 기동 시 `RuntimeError`

#### 5.8 품질 관리

- `pytest` → `tests/test_pipeline.py`: 합성 CSV(클래스별 2파일 + 테스트 2파일, 2시간, gap·캡 드레싱·종료 코드 포함)를 임시 폴더에서 2→3→4→4'→`--score`→5단계(`/predict`)까지 돌리고, API 점수가 오프라인 점수와 같은지 확인한다. 실데이터 폴더는 건드리지 않는다. 파이프라인 코드를 고치면 이것부터 돌린다.
- `ruff check .` 통과 상태. `black .`은 `pyproject.toml` 설정(120자)을 따른다.
- 실데이터 파이프라인을 다시 돌려야 하는 변경: 전처리 규칙(2단계 코드) → 2·3·4단계 전부. 모델·윈도우만 → 4단계. 서빙만 → 재실행 불필요(`pytest`로 확인).

#### 5.9 진행 이력

| 날짜 | 내용 |
|---|---|
| 09-17 ~ 09-18 | 논문 리뷰, 데이터 확보, 원본 CSV 점검(`file_check.py`, `error_check.py`), 접근 방식 결정(1-step 이상탐지, 2절) |
| 09-22 | 벤치마크 코드 리뷰·수정(5.2), EDA(5.3), 전처리(5.4), IsolationForest 베이스라인(5.6), FastAPI 서빙(5.7), `MEMORY.md` 작성 |
| 09-23 | 테스트 8파일 전처리·평가 추가(5.5, 5.6). 파이프라인 점검: gap 채움 버그(ffill+bfill로 120초까지 메꿔짐) 수정 후 전처리·재학습, 번들에 전처리 설정 포함·서빙이 읽도록 변경, E2E 스모크 테스트·`requirements.txt`·`pyproject.toml` 추가 |

#### 5.10 현재 한계와 다음 단계

1. 베이스라인은 종료 상태(E029 등)조차 임계값 아래로 본다. 시도할 것: `--scale per-file --drop-static`(건 간 오프셋 제거 — 단 서빙용 스케일 전략을 같이 정해야 함), 윈도우 확대(5~10분), 시퀀스 모델(LSTM-AE 등), `label`을 쓰는 지도학습 비교.
2. E01_0은 고장 직전 1시간의 85%가 `non_welding` — 고장이 정비 중 발생하는 패턴인지 72파일에서 확인해야 하며, 그렇다면 `non_welding`을 입력에서 빼야 모델이 플래그만 외우지 않는다.
3. test_2·test_3은 행의 87~95%가 `non_welding`(학습 최대 45%) — 이 두 건에서 c16 규칙이 "캡 드레싱"이 맞는지 c16 분포를 확인해야 한다. 테스트 E02·E04 지표와 오탐(하루 7~10회 지속 알람)은 여기서 나온다.
4. 서빙 윈도우는 트레일링 60초, 학습 윈도우는 분 경계 정렬 — 정확히 맞추려면 서빙도 분 경계로 자른다.
5. `Benchmark_fixed.py`의 실제 학습 경로는 미검증.
6. 3절의 하류(온톨로지/RAG 원인 분석, LLM 리포트, Slack)는 미구현. 입력 계약은 5.7의 `AnomalyResult`.
