# GelSLAM 시뮬레이션 실행 점검

**점검일 2026년 10월 21일 · 대상 rock figure8 · 현재 소스 재생 및 접촉 판정 비교**

## 핵심 결과

**GT상 연속 접촉인 구간에서 접촉을 놓치는 현상을 재현했습니다.** SDK 접촉 마스크 뒤에 적용되는 NormalFlow 침식이 일부 영상을 비접촉으로 바꿉니다. 이후 접촉 정합 실패와 IMU 복구 예측이 이어지고, 위치 오차가 남습니다.

접촉 존재 판정만 침식 전 마스크로 바꾼 진단에서는 공통 시각 1,358개의 위치 RMSE가 **7.22 mm에서 2.96 mm로 감소**했습니다. 같은 bag, 초기 상태, 평가 정렬을 사용했습니다. 정합용 마스크와 제품 소스는 변경하지 않았습니다.

| 항목 | 현재 경로 | 진단 경로 |
| --- | ---: | ---: |
| 공통 시각의 위치 RMSE | **7.223 mm** | **2.964 mm** |
| 공통 시각 수 | 1,358개 | 1,358개 |
| 전체 유효 상태의 위치 RMSE | 7.211 mm | 2.999 mm |
| 전체 유효 상태 수 | 1,369개 | 1,396개 |
| 회전 RMSE | 0.108° | 0.105° |
| IMU 원점 속도 RMSE | 3.302 mm/s | 3.104 mm/s |
| GT 접촉 중 비접촉 모드 영상 | 53개 | 13개 |
| IMU 복구 키프레임 | 4개 | 1개 |
| 초기 공유점 부족 실패 | 66회 | 23회 |

전체 유효 상태 수가 다르므로 직접 비교에는 공통 시각의 RMSE를 사용합니다. 해당 위치 RMSE 감소율은 약 **59%**입니다. 한 시뮬레이션의 진단 결과이며, 실측 적용 성능은 별도로 검증해야 합니다.

![시간별 위치 오차와 접촉 픽셀 수](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/simulation_diagnosis.png)

파란색은 현재 경로입니다. 주황색은 초기화 후 접촉 존재만 침식 전 마스크로 판정한 진단 경로입니다. 위쪽 오차 곡선의 빈 구간은 유효 상태가 없는 시각입니다. 아래쪽 점선은 접촉 판정 기준인 300픽셀입니다.

## 재현 조건

| 항목 | 확인한 조건 |
| --- | --- |
| 입력 | `fixture-rock-long-figure8-20261001-v1-gelslam-image` |
| 영상 / IMU | 1,501개 / 15,001개 |
| 입력 시간 범위 | 1.0~76.0 s |
| IMU 원본 간격 | 모두 정확히 5 ms |
| IMU 시각 중복 / 역행 | 0개 / 0개 |
| 설정 | 현재 소스 `bag_simulation_config.yaml`의 복사본 |
| 적분 | `midpoint` |
| 프론트엔드 | IMU와 normal/height 잔차의 공동 최적화 |
| 초기화 | 정지 구간 1.0~2.9 s에서 시작하여 첫 수락 접촉 4.35 s로 전파 |
| GT 초기 자세와 속도 옵션 | 정지 초기화가 덮어씀 |
| 이상 IMU 대체 | 꺼짐 |
| 백엔드 최적화 / 피드백 | 모두 꺼짐. 새 재생에서는 백엔드 객체도 생성하지 않음 |
| 평가 정렬 | 4.35 s의 첫 상태를 기준으로 한 고정 SE(3) 변환 |
| 입력 순서 | 모든 영상을 순서대로 처리하고, 끝점 보간용 IMU 샘플 한 개까지 제공 |

GT는 평가에 사용했습니다. 접촉 판정 진단에도 GT 마스크를 넣지 않았습니다. 재생은 처리 지연이나 ROS callback 순서를 검증하는 실시간 시험이 아닙니다.

### 저장 실행과의 일치

저장된 `rock_figure8_VIO2`는 2026년 10월 21일 11시 53분의 실행입니다. 프론트엔드 로그는 피드백 비활성화를, 백엔드 로그는 최적화와 피드백이 꺼진 저장 경로를 기록합니다.

현재 소스로 재생한 기본 경로는 저장 실행과 **1,369개 상태의 시각·종류·pose·속도가 모두 일치**했습니다. 저장된 `State`의 pose와 속도 차이 최댓값은 각각 0이었습니다. ROS 메시지를 거친 종료 pose CSV에는 자료형 차이에 따른 작은 차이가 있습니다.

따라서 아래 접촉 판정과 오차 증가는 현재 소스에서 재현된 현상입니다.

근거: [재생 검증](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/verification.json), [현재 경로 결과](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/current_midpoint/result.json), [저장 프론트엔드 로그](/home/dongjae/.ros/log/python_2535735_1792551214919.log), [저장 백엔드 로그](/home/dongjae/.ros/log/python_2535737_1792551214055.log)

## 비접촉 모드와 GT 접촉의 차이

이 데이터는 반복적인 lift를 포함한 bag이 아닙니다. 데이터 manifest는 `intermittent_contact=false`, `planned_contact_interruptions=0`을 기록합니다. GT 접촉 마스크도 4.3~73.85 s에 연속적인 접촉 지지를 나타냅니다.

GT 접촉 영상 1,392개에서 마스크의 최소 지지는 2,659픽셀이었습니다. 같은 시각의 추정기 `no_contact_coasting`은 실제 lift의 증거가 아닙니다.

| GT상 접촉 구간의 추정기 비접촉 모드 | 영상 수 |
| --- | ---: |
| 4.30 s | 1개 |
| 19.85~20.45 s | 13개 |
| 37.10~37.65 s | 12개 |
| 54.25 s | 1개 |
| 54.35~55.00 s | 14개 |
| 71.45~71.95 s | 11개 |
| 73.85 s | 1개 |
| 합계 | **53개** |

각 구간의 접촉 픽셀 수를 SDK와 NormalFlow 경로로 다시 계산했습니다. 최종 판정은 재생의 비접촉 모드와 모든 검사 영상에서 일치했습니다.

근거: [입력 시각과 GT 접촉](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/input_timing_contact.json), [데이터 manifest](/home/dongjae/dataset/GeLISLAM/fixture-rock-long-figure8-20261001-v1-gelslam-image/source_manifest.json), [접촉 검사](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/contact_analysis.json)

## 접촉 마스크 처리에서 확인한 원인

현재 경로는 다음 순서로 접촉을 판정합니다.

1. SDK가 배경과의 색 차이 및 복원 높이로 접촉 마스크를 만듭니다.
2. SDK의 색 마스크에는 7×7 팽창과 15×15 침식이 적용됩니다.
3. NormalFlow `Frame`이 최종 SDK 마스크에 5×5 침식을 추가합니다.
4. 침식 후 픽셀이 300개 이상일 때 접촉으로 판정합니다.

SDK의 높이 기준은 0.2 mm입니다. 실제 접촉 지지가 넓어도 복원 마스크가 얇거나 작은 조각이면 추가 침식으로 지지가 크게 줄어듭니다.

| 영상 시각 | GT 접촉 픽셀 | SDK 마스크 | Frame 침식 후 | 현재 판정 |
| --- | ---: | ---: | ---: | --- |
| 19.80 s | 9,178 | 1,353 | 546 | 접촉 |
| 20.00 s | 8,391 | 287 | 13 | 비접촉 |
| 20.50 s | 4,860 | 823 | 478 | 접촉 |
| 54.20 s | 9,862 | 1,205 | 487 | 접촉 |
| 54.60 s | 6,724 | 184 | 70 | 비접촉 |
| 55.05 s | 6,353 | 688 | 344 | 접촉 |

GT상 접촉인 영상 중 SDK 마스크부터 300픽셀 미만인 영상은 **12개**였습니다. SDK 마스크는 300개 이상이지만 추가 침식 뒤 300개 미만이 된 영상은 **41개**였습니다. 두 경우를 합하면 현재 비접촉 판정 53개가 됩니다.

![영상과 침식 전후 접촉 마스크](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/contact_mask_examples.png)

높이 기준만 낮춰도 문제가 모두 사라지지는 않았습니다. 같은 색 마스크와 Frame 침식을 유지한 접촉 판정 계산 결과는 다음과 같습니다. 이 표는 마스크 검사이며 전체 SLAM 재생 결과가 아닙니다.

| 높이 기준 | GT 접촉을 놓친 영상 | GT 비접촉에서의 접촉 판정 |
| --- | ---: | ---: |
| 0.20 mm | 53개 | 0개 |
| 0.15 mm | 44개 | 0개 |
| 0.10 mm | 35개 | 0개 |
| 0.05 mm | 34개 | 0개 |

근거: [마스크별 픽셀 수](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/contact_counts.csv), [SDK 접촉 처리](/home/dongjae/miniconda3/envs/gel-slam/lib/python3.10/site-packages/gs_sdk/gs_reconstruct.py:80), [Frame의 접촉 판정](/home/dongjae/miniconda3/envs/gel-slam/lib/python3.10/site-packages/normalflow/utils.py:10), [추가 침식](/home/dongjae/miniconda3/envs/gel-slam/lib/python3.10/site-packages/normalflow/utils.py:128)

## 위치 오차가 남는 과정

첫 문제 구간 전 4.35~19.8 s의 위치 RMSE는 **0.162 mm**였습니다. 이후 접촉 판정이 끊기는 시점과 위치 오차 증가가 반복됩니다.

| 마지막 수락 접촉 상태 | IMU 복구 예측 상태 | 예측 구간 | 구간 시작 위치 오차 | 복구 예측 위치 오차 | 다음 접촉 최적화 위치 오차 |
| --- | --- | ---: | ---: | ---: | ---: |
| 19.80 s | 21.00 s | 1.20 s | 0.254 mm | 1.479 mm | 21.05 s에서 1.394 mm |
| 37.05 s | 38.20 s | 1.15 s | 1.236 mm | 4.149 mm | 38.25 s에서 3.900 mm |
| 54.20 s | 55.55 s | 1.35 s | 4.862 mm | 13.132 mm | 55.60 s에서 12.556 mm |
| 71.40 s | 72.50 s | 1.10 s | 13.318 mm | 13.192 mm | 72.55 s에서 13.189 mm |

위의 복구 상태는 접촉 측정으로 최적화한 상태가 아니라, 마지막 수락 상태에서 전파한 **IMU 예측**입니다. 비접촉 판정 뒤 접촉이 다시 검출돼도 기존 참조와의 초기 공유점이 10개 미만인 실패가 반복됩니다. 현재 경로는 실패가 지속되면 그 영상에 IMU 예측 pose를 부여하고 새 참조 키프레임으로 사용합니다.

그 다음 영상에서 상대 접촉 정합이 성공해도, 이전 참조로 돌아가는 절대 위치 측정이 생기는 것은 아닙니다. 이 프론트엔드 전용 실행에서는 이미 누적된 위치 오차가 남는 패턴을 보였습니다.

55.55 s의 복구 예측까지는 구간 시작 대비 오차 벡터가 약 9.17 mm 변했습니다. 이때 속도 오차는 9.86 mm/s였습니다. 전체 위치 오차의 축별 RMSE는 world X 3.46 mm, Y 6.13 mm, Z 1.56 mm였습니다. 회전 RMSE 0.108°만 보고 위치 추정도 충분히 정확하다고 판단할 수는 없습니다.

`max_coast_duration_s=1.0`은 비접촉 coasting의 제한입니다. 위 복구 예측 경로는 `force=True`로 별도 실행됩니다. 따라서 실제 복구의 IMU 구간이 모두 1초 이내라고 해석하면 안 됩니다.

근거: [상태별 오차](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/current_midpoint_errors.csv), [복구 이벤트](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/verification.json), [비접촉 판정 후 전파](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:2527), [IMU 복구와 참조 교체](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:1114)

## 접촉 판정만 바꾼 진단

진단에서는 초기화 후 `frame.is_contacted`만 SDK 마스크의 픽셀 수로 판정했습니다. `Frame.C`의 침식, normal/height 정합, IMU 입력과 가중치, 실패 복구 정책은 유지했습니다.

첫 수락 접촉 시각은 양쪽 모두 4.35 s입니다. 첫 상태의 pose·속도·bias도 일치합니다. 입력 SHA256과 평가 정렬 변환도 일치합니다. 공통 시각 1,358개의 위치 RMSE가 7.223 mm에서 2.964 mm로 줄었습니다.

GT 접촉 영상의 비접촉 모드는 53개에서 13개로 줄었습니다. 초기화 판정을 유지했으므로 4.30 s의 비접촉 판정도 유지됩니다. IMU 복구 키프레임은 4개에서 1개로 줄었습니다.

별도의 예비 진단에서는 초기 접촉도 4.30 s로 바뀌었습니다. 그 결과는 초기화 차이가 있어 주 비교에서 제외했습니다. 보고서의 비교 수치는 초기화를 고정한 진단 결과입니다.

이 결과는 **접촉 존재 판정과 정합용 침식의 결합을 먼저 수정할 근거**입니다. 남은 약 3 mm 오차와 첫 복구는 별도 검증이 필요합니다. 단순히 침식을 제거하는 제품 변경을 수행한 것은 아닙니다.

근거: [진단 실행 결과](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/presence_after_initialization/result.json), [공통 시각 비교](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/analysis.json), [초기 상태와 입력 일치 검증](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/verification.json)

## 다음 수정 순서

1. **접촉 존재와 정합 가능 여부를 분리합니다.** 침식된 픽셀 수가 적다는 이유만으로 실제 비접촉으로 취급하지 않도록 합니다.
2. **정합 지지가 줄어드는 구간을 기록합니다.** SDK 마스크, Frame 마스크, 공유점 수, 참조 시각을 함께 남깁니다.
3. **참조 교체 후의 위치 오차를 검증합니다.** IMU 예측으로 만든 키프레임과 다음 접촉 최적화를 구분합니다.
4. **남은 속도·bias·prior 오차를 조사합니다.** 접촉 판정 수정 뒤에도 남는 약 3 mm 오차를 같은 공통 시각으로 비교합니다.
5. **실제 lift가 있는 별도 bag으로 확장합니다.** 이 연속 접촉 bag의 복구 결과를 실제 비접촉 운용 성능으로 사용하지 않습니다.

## 변경 파일과 검증 자료

제품 소스와 설정은 변경하지 않았습니다. 이 보고서와 `diagnostics/simulation_review_20261021`의 분석·재생 자료만 추가했습니다.

| 파일 | 내용 |
| --- | --- |
| [재생 도구](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/replay_instrumented.py) | 기존 순차 재생 도구를 사용한 관찰 및 임시 접촉 판정 |
| [입력 설정 복사본](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/input_config.yaml) | 재생에 사용한 현재 시뮬레이션 설정 |
| [재생 검증](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/verification.json) | 입력 해시, 상태 일치, 초기화 통제, 복구 이벤트 |
| [분석 결과](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/analysis.json) | 상태 종류별·시간 구간별·공통 시각별 오차 |
| [접촉 분석](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/contact_analysis.json) | 침식 전후 판정과 높이 기준 검사 |
| [기본 경로 로그](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/replay_console.log) | 현재 소스 1,501개 영상 재생 |
| [초기화 통제 진단 로그](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/presence_controlled_console.log) | 초기화 후 접촉 판정만 바꾼 재생 |

과거 적분 방식 비교와 로컬 VI·피드백 실험도 확인했습니다. 일부 기록은 현재와 다른 tracker·solver 소스를 사용했습니다. 해당 기록의 발산을 현재 소스의 발산으로 단정하지 않았습니다.
