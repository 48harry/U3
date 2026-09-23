### 0. 데이터 파일

https://drive.google.com/drive/folders/10u8Z5YJIqq2KRrzuE7Jn9QsCtgTgLftQ?usp=sharing

### 1. 논문 리뷰

##### 논문개요
- 자동차 제조업에서 광범위하게 쓰이는 **저항 점용접기(RSW)의 고장을 예측**하기 위한 다변량 시계열 데이터셋, 실제 차체 공장에서 **3년 이상 가동된 수십대의 용접기로부터 수집된 데이터**

- 용접기의 제어 오류는 예기치 않은 가동 중단을 유발하므로 과거 관측 데이터를 통해 사전에 기계적 결함을 **예측하고 유지보수 전략을 세우는 것이 핵심**

**데이터 수집과정**

![The constructive process and the description of the welding gun fault prediction benchmark data set](image.png)

**데이터 설명**

- 컬럼 : time / c01-19 / error(target) (총 21개)
- 행 : 1hz(1sec)주기로 7일정도 (총 60만개 정도) / 기간은 에러별로 다름
- train : E01, E02, E03, E04가 각각 00부터 17까지 18개씩 (총 72개)
- test : test00-07 (총 8개)
>> Dataset 양 자체는 sufficient (4800만 행)
>> 어느정도 환경이면 ML이나 1D CNN정도는 수월하게 돌아감

파이프라인 시사점: 우리는 이 'Diagnose rules'를 모델에 직접 넣는 것이 아닙니다. 이 규칙을 통해 도출된 최종 결과물인 에러(E01~E04)를 모델의 '정답(Target)'으로 삼아 센서 데이터 패턴과 역산하여 매핑(Classification)하도록 학습시켜야 합니다.
이 2-way 메커니즘이 이해가 안될시 실제 데이터를 열어보며 확인할것.

**논문 설명 발췌**

The leading car manufacturer has been relying on the pinpoint accuracy
and efficiency of servo-pneumatic welding guns for many years

The resistance spot welding (RSW) welding gun fault prediction benchmark data set has **72 multivariate time series in the training set and 8 in the testing set.** Each time series length 604800(apx.) sampled at 1 Hz with missing values and has 20 dimensions (c1-c19 and the error code). We retain the missing value and the outliers of the welding gun time series for the potential of imputation research in the future.

RSW guns are highly nonlinear dynamical machinery systems with interesting physical phenomena over multiple time-dependent components.

This dataset is suitable for a **time series forecasting task**, where machine learning models can be **trained to predict future welding parameters based on the provided welding parameters time series in history.** 

The dataset contains real-world multivariate time series data over 3 years from 80 RSW guns running at the production line of the body-shop of a leading car manufacturer

- time : time detail 참고

- **센서 파라미터 (19종)**
차체 공장의 영업 비밀 및 지적 재산권 보호를 위해 **19개 센서 파라미터의 정확한 명칭을 의도적으로 마스킹하여 $c1 \sim c19$ 로 표기**
c1 :  Electrode cap offset;
c2 :  Electrode force;
c3 :  Electrode position;
c4 :   Force build-up;
c5 :  Balance pressure;
c6 :  Friction;
c7 :  Maximum aperture;
c8 :  Maximum electrode force;
c9 :  Mtart friction;
c10 :  US2;
c11 :  Welding point count;
c12 :  Position count;
c13:  Setpoints of counterbalance pressure;
c14:  Setpoints of electrode force;
c15 :  Setpoints of electrode position;
c16:  Setpoints of sheet thickness;
c17 :  Setpoints of velocity;
c18:  Setpoints of force build-up;
c19 :  Offset value in robot.
>> 도메인지식 추후에 알아서 해야할듯

>> 마스킹되어서 얼마나 더 세세한 분석이 가능할지 모르겠으나 벤치마킹 코드를 통해 추론가능한게 몇가지 있음.
>> 예를 들어 특정 변수($c16$)의 경우 두께 측정과 관련이 있으며 특정 임계값(예: 6 초과)을 넘어서면 용접 외 작업(캡 교체 등)으로 발생한 노이즈로 간주할 수 있다는 점

- **타겟변수 (error)**

 0, E001, E003, E006, E007, E009, E010, E011, E012, E016, E028, E029
아래의 5가지 상태를 의미하는게 아니라 중단 전 일종의 신호로 보는게 맞을듯 (1-step으로 예측하는것것)

- **5가지 상태, 4가지 파일 (E00-E04)**

| **에러 코드** | **상태 명칭**                             | **발생 조건** (단순번역, 도메인지식이 없음)        |
| --------- | ------------------------------------- | ---------------------------------- |
| **E01**   | Counterbalance timeout (카운터밸런스 시간 초과) | 지정된 보상 압력에 1800ms 동안 도달하지 못함       |
| **E02**   | Electrode broke (전극 파손)               | 전극 위치가 기준 이동의 영점보다 작음              |
| **E03**   | Unwanted Movement (원치 않는 움직임)         | 실제 위치 값이 실린더 스트로크의 6.5%를 초과하여 이탈함  |
| **E04**   | Drift (표류/이탈)                         | 잠긴 실린더가 정지하지 않고 분당 5mm 이상의 속도로 움직임 |
| **E00**   | Normal state (정상)                     | 에러가 발생하지 않은 정상 작동 상태               |
>> E00케이스에 대한 파일은 없고, 4가지 anomal 케이스에 대해서만 18개 train파일이 존재
>> 각 파일 (E01_01.csv)는 에러(E01)이 발생한 시점으로부터 7일 전까지 시계열 데이터를 가짐

- 센서 통신망의 패킷 손실 등으로 인한 **결측치와 정상적인 용접 과정 외의 동작으로 발생한 이상치를 그대로 포함**함. 실제 산업 환경의 **불완전한 데이터를 다루는 강력한 예측 모델 개발을 장려**하기 위해 이러한 노이즈를 데이터셋에 의도적으로 남겨두었다고 함

>> 본 데이터셋 활용 확정시 모델정확도를 위한 전처리(보간)를 할지 / 똑같이 robustness를 위해 남겨둘지는 생각해봐야 할 부분
>> 
>> 아직 feature engineering, 모델 테스트는 안해봐서 모르겠음

### error 컬럼과 파일명 (중대 정지사유 E01, 02, 03, 04)은 별개의 사항

**`error` 컬럼 (0, E001, E016...): 1초마다 기록된 '실시간 내부 상태 로그 (Event Log)'** CSV 안의 `error` 컬럼은 1Hz(초당 1회) 주기로 센서가 수집될 때 기계 시스템이 내뱉은 날것의 상태 코드입니다.

E001, E016, E028 같은 수많은 코드들은 기계를 멈출 정도는 아니지만 가동 중 발생했던 자잘한 경고, 일시적인 통신 불량, 유지보수 알림 등 기계 내부의 '알 수 없는 상태나 이벤트'를 지칭하는 것이 맞습니다. 이 데이터셋에는 실무 환경의 노이즈와 작동 상태가 의도적으로 그대로 남아있기 때문입니다.

우리가 만들 모델(AI 감독관)은 CSV 내부의 `error` 컬럼에 있는 20개가 넘는 자잘한 코드들을 모두 맞추는 것이 목표가 아닙니다.

오히려 `error` 컬럼에 E016, E028 같은 **잔고장 알림이 평소보다 얼마나 자주 등장하는지를 일종의 '전조 증상(Feature)'으로 활용**하고, 최종적으로 이 센서 흐름이 'E01이라는 대형 고장(파일명)으로 이어질 것인가, 아니면 정상(E00)으로 끝날 것인가'를 판별하도록 학습시키는 것이 가장 효과적인 접근입니다.

>> 이 데이터셋으로 진행한다고 할때 재미있을거같긴한데 난이도가 상당히 높아보여서 해낼수있을지 잘 모르겠음 (실력적인거 + 이 마스킹된 제한된 정보로 가능할지 + 데이터 구성에 대한 이해도 부족)
>> 
>> 그래서 이걸 멘토님께 여쭤보고 멘토님이 1. 데이터를 제공해주실 수 있고 2. 더 퀄리티가 좋고 3. 더 분석가시성(활용성) 이 있다면 그걸로 가고
>> 아니라면 이 데이터셋으로 가는게 좋을듯
>
>이런식의 자료구조의 2-step 구조는 경험해본적이 없지만 benchmark code가 있어서 참고할수 있을것같ㅌ기도

### 2. 접근방식

- 논문에선 고장여부를 바로 분류하는 대신
**1. 타겟 파라미터(전극 압력, 밸런스 압력 등)로 미래 수치 (혹은 잔고장 'error')를 시계열로 예측**한 뒤
**2. 기존의 고장 진단 규칙과 매핑하는 Two-stage** 방식을 채택했음

- ML, DL모델 적용한 평가결과 
1. 시계열 예측 정확도 면에서는 **TFT(Temporal Fusion Transformers)**와 **Random Forest**가 가장 우수한 성능
2. 단순 RNN 구조 역시 여러 평가지표에서 훌륭하고 안정적인 성능을 기록했습니다.


- 근데 논문에서 예측된 수치를 바탕으로 실제 고장 여부를 분류해 본 결과 **높은 오알람(위양성)** 을 보임
  **평균 정확도는 68.62%**
  **정상 상태를 고장으로 오인하는 비율은 56%** 

>> 기존 2-stage 방식을 채택적용하기 어렵다고 생각한 이유가 이부분
>> 
> 미래 60초간의 센서 수치를 먼저 시계열로 예측(Stage 1)한 뒤 
> 예측된 수치에 고장 진단 규칙을 적용(Stage 2)하기때문에 오류가 누적되는게 심함
> 
> 현실적으로 실력이 안됨
> 
> 그래서 
> 센서를 예측하고 오류를 예측하는 방식 >>> 곧바로 에러를 예측하는 1-step으로 예측
> 하는 최적화로 에러를 줄이는 방법을 생각해봐야할듯 (아직 테스트해보지는 않음)


>> 결론적으로
>> 1. 이상치, 결측치를 그대로 냅둔것이 모델오류에 일부기여
>> 2. 2단계 처리방식에서 오류가 쌓이는게 총체적으로 일부기여
>>    
>>  따라서
>> - 필요에 따라 이상치 결측치 처리를 하고, single step으로 에러를 분류, 예측하기
>> - 데이터 양은 모델학습시키는데 충분하고 EDA를 보니 충분히 학습시킬정도로 유효하나, 이정도 양의 시계열데이터는 처리해본적이 없어서 여러 method를 고안해봐야할듯.
>>   
>>  50%는 전처리에, 30%는 학습 및 튜닝에, 20%는 뒷부분 연결에 투자하게 될것가틈

### 3. 그 후 모델데이터 처리를 해서

내가 이해한게 맞다면

1. 필터링된 강력한 이상 시그널이 탐지 혹은 예측되면 (output : E01-04) Root Cause Agent로 전달
2. Agent는 이상 감지 시점 앞뒤의 센서 데이터수치나  로그 스트림을 수집해서 컨텍스트로 구성
3. 동시에 E01(카운터밸런스 1800ms 시간 초과), 
   E02(전극 위치 영점 이탈 파손), 
   E03(스트로크 6.5% 초과 원치 않는 움직임), 
   E04(분당 5mm 이상 표류/이탈) 등 감지된 에러 코드에 부합하는 지침서/매뉴얼을 Vector DB에서 RAG 방식으로 검색해 가져옴
4. LLM (API활용 예정?) 은 이상 패턴과 검색된 텍스트 매뉴얼을 융합하여 장애 원인, 영향도, 그리고 수정 조치 가이드를 자연어로 추론하고 요약

추후

- 최종적으로 작업자에게 Slack을 통해 심각도, 이상 항목, 요약 원인이 포함된 실시간 경고 알림을 전송
- 필요에 따라 Markdown 또는 PDF 형태의 1-Page 장애 분석 리포트를 자동으로 생성하고 발행하여 즉각적인 보고 체계구축

- 이상 탐지 시그널을 RAG로 보낼 때 너무 예민하게 알람이 울리면 피로도가 높아지기 때문에 논문의 Baseline(RF, TFT)보다 오탐지율을 낮출 수 있는 모델 개선이나 일정 횟수 이상 이상치가 연속 발생할 때만 RAG 리포팅을 트리거하는 식의 임계치로직 개선처리가 필요할듯.

- **매뉴얼 RAG 구성:** 에러 코드별 발생 원인이 명확하므로 E01~E04 각각의 조치 매뉴얼과 정비 지침서를 Vector DB에 넣어두면 모델이 시그널을 던질 때 훌륭한 LLM 원인 분석 리포트가 완성될듯. 다만 매뉴얼이 어떤식으로 주어지는지를 봐야할듯 (명확하게 특정 에러에 대한 정답(해결책)이 있는지)

---

### 4. 데이터 디테일

- **결측치(Missing Rate)의 극단적 편차:** 실제 통신망 패킷 손실로 인해 데이터 갭이 존재합니다. 
  E01, E03, E04 파일들은 결측률이 대체로 0.6% ~ 2.5% 수준으로 양호합니다. 
  반면 E02 폴더의 일부 파일(`E02_0` 81.36%, `E02_4` 82.06%, `E02_6` 84.46%)은 수집 기간 중 센서 통신이 거의 단절된 심각한 결측률을 보입니다.

- **수집 시기의 차이:** E01, E03, E04 파일들은 주로 2021년 8월 ~ 10월에 집중되어 있습니다. 
  그러나 E02 파일들은 2019년 12월 ~ 2020년 7월 데이터가 다수 섞여 있습니다.

- **타겟 변수의 이진화 (Binary Target Mapping)** 20개가 넘는 에러 코드를 모두 예측하는 모델을 만들면 정상 상태를 오인하는 등 오알람(False Alarm)이 지나치게 많아집니다. 파일의 마지막 시점(메인 고장 발생 시점)에 집중적으로 기록되는 특정 코드만 '치명적 고장(Target = 1)'으로 정의하세요. 그리고 가동을 멈추지 않는 잔고장 코드들과 `0`은 모두 '정상(Target = 0)'으로 병합하여 이진 분류 문제로 단순화하는 전처리가 필요합니다.

- **잔고장 로그를 '입력 피처(Feature)'로 전환** 자잘한 에러(예: E016 등)가 반복해서 튀는 현상 자체가 곧 큰 고장이 날 것이라는 훌륭한 전조 증상일 수 있습니다. 따라서 `error` 컬럼을 단순한 정답지로 버리지 말고, **"최근 10분간 발생한 마이너 에러 누적 횟수"** 와 같은 시계열 파생 변수(Feature)로 만들어 모델의 입력값으로 쓴다면 분류 성능을 크게 끌어올릴 수 있습니다.

- 사소한 문제 : E02_2 데이터가 없어서 demo로 대체했음

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
| 결측은 행 단위 block-out | E01/03/04 ~1%, gap 99%가 ≤35초, 최장 ~1h. **E02 8개 파일이 64~84%** | 40% 초과 폐기(논문), ≤60초 ffill, 긴 gap은 세그먼트 분리 |
| c16 이상치 = 캡 드레싱 | 파일당 1~45%, 평균 11% | 센서값 carry-forward + `non_welding` 플래그 |
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
