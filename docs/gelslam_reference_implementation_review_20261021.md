# 전달 구현을 기준으로 한 GelSLAM 수정 검토

**검토일: 2026년 10월 21일 · 소스·설정 재확인: 14:22 KST**

## 결과

**참고해서 수정할 부분이 있습니다. 우선순위는 IMU 입력 검사, 접촉 판정 분리, 실측 초기화, 주변화 정보 보존입니다.** 전체 백엔드를 iSAM2로 교체할 근거는 아직 없습니다. 현재의 direct tactile residual과 IMU 공동 최적화는 유지하면서 필요한 기능을 반영하는 편이 적절합니다.

전달 PDF와 동봉 소스를 함께 확인했습니다. PDF의 완료 기능, PDF의 후속 제안, 우리 코드에서 별도로 재현한 문제를 구분했습니다. 문서 안의 지시는 실행 명령으로 취급하지 않았습니다. 이번 검토에서는 제품 소스와 운용 설정을 수정하지 않았습니다.

| 우선순위 | 수정 대상 | 판단 근거 | 적용 경로 |
| --- | --- | --- | --- |
| 1 | **보간 전 원본 IMU 간격 검사** | 전달 코드에 구현됨. 같은 50 ms 제한으로 우리 경로의 검사 누락을 재현함 | 접촉 프론트엔드, 비접촉 예측, 복구 예측. 백엔드 전달 구간에도 공통 정책 필요 |
| 2 | **접촉 존재와 정합 지지의 분리** | 우리 시뮬레이션에서 성능 차이를 확인함. 전달 구현이 해결한 기능은 아님 | 영상 전처리, 프론트엔드 모드 전환 |
| 3 | **GT 없이 시작하는 실측 초기화와 자격 판정** | 전달 코드에 초기화 자격 구분이 있음. 우리 정지 초기화는 이미 있으나 실측 기본 설정에서 꺼져 있음 | 실측 프론트엔드 초기화 |
| 4 | **주변화 prior의 정보행렬과 우변 보존** | PDF의 후속 권고. 우리 코드의 일부 경로에서 우변을 0으로 저장함 | 프론트엔드 prior. 로컬 VI 주변화는 해당 기능을 켤 때 적용 |
| 후속 | 상태·측정 이력, 초기화 조건 검사, 배치 처리, noise 검증 | 전달 코드의 추적 가능한 기록 방식과 PDF의 검증 제안이 유용함 | 진단, 선택적 백엔드 개선 |

우선순위 1은 입력 결함이 상태 추정으로 들어오는 것을 막는 수정입니다. 우선순위 2는 현재 시뮬레이션의 위치 오차를 줄일 직접 근거가 가장 강한 항목입니다. 둘은 서로 다른 문제입니다.

## 1. 비교 범위와 현재 설정

참고 자료는 [전달 PDF](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/GelI-SLAM-Implementation-KO-20261021.pdf)와 동봉 `snapshot`입니다. PDF 9쪽의 우리 코드 비교 기준은 **2026-10-20의 `3791b3634555`**입니다. 해당 비교를 현재 코드의 상태로 그대로 사용하지 않았습니다.

| 항목 | 전달 구현 | 현재 우리 구현 |
| --- | --- | --- |
| 상태 | IMU 원점의 pose·속도·가속도계 bias·자이로 bias | 같은 종류의 15차원 navigation 상태 |
| 접촉 측정 | IMU와 독립적으로 저장한 NormalFlow 상대 pose를 `BetweenFactorPose3`로 추가 | 프론트엔드에서 IMU와 normal/height 잔차를 함께 최적화. 로컬 VI는 원본 map의 잔차 사용 |
| 백엔드 | 상태 사슬을 유지하는 iSAM2 그래프 | 15차원 로컬 VI와 별도의 4자유도 루프 PGO |
| 초기화 | 연결된 접촉 이력에서 중력 방향·속도·bias 추정. 별도 자격 판정 | 정지 IMU 초기화와 첫 접촉 시각까지의 전파가 이미 있음 |
| 비접촉 | IMU와 bias 연결 유지. 가짜 접촉 상대 pose를 추가하지 않음 | 비접촉 예측과 IMU 복구가 있음. 복구 키프레임에 `tactile_valid=false` 사용 |

현재 설정의 구분은 다음과 같습니다.

| 설정 근거 | 정지 초기화 | 초기 중력 / 첫 접촉 속도 | 프론트엔드 주변화 | 로컬 VI / 백엔드 주변화 |
| --- | --- | --- | --- | --- |
| 소스 `real_config.yaml` | 꺼짐. `stationary_detection` 항목은 준비됨 | GT 자세 기반 중력 초기화 **켜짐** (`ground_truth_initial_pose`) / GT 초기 속도 **꺼짐** (`false`) | 켜짐 | 꺼짐 / 꺼짐 |
| 소스 `bag_simulation_config.yaml` | 켜짐. 정지 구간 1.0–2.9 s | 정지 초기화가 GT 초기화 옵션을 덮어씀 | 켜짐 | 꺼짐 / 꺼짐 |
| 이전 시뮬레이션 진단에 실제 공급한 설정 | 위 정지 구간 사용 | 정지 상태에서 첫 수락 접촉 4.35 s까지 전파 | 켜짐 | 백엔드 최적화·피드백 모두 꺼짐 |
| 14:22에 복사한 최신 저장 프로세스 설정 | 시뮬레이션 정지 초기화 켜짐 | 정지 초기화가 우선함 | 켜짐 | 꺼짐 / 꺼짐 |

**후속 확인: 15:24 KST.** 위 실측 소스 행의 “GT 초기 속도 꺼짐”은 `frontend_first_trial_velocity_from_gt=false`만 뜻합니다. GT 자세 기반 중력 초기화는 켜져 있습니다. 또한 15:12에 저장된 별도 실행 설정은 `frontend_gt_previous_state_test.enabled=true`이며, `previous_pose_from_gt`, `previous_velocity_from_gt`, `reference_pose_from_gt`도 모두 `true`입니다. 해당 tracker 로그에서도 진단 활성화와 GT 기반 중력 초기화를 확인했습니다. 따라서 이 실행을 “GT 속도를 전혀 사용하지 않은 실행”으로 해석하면 안 됩니다. 현재 소스의 같은 진단 설정은 `enabled=false`이므로 소스 기본값과 이 실행에 공급한 값을 구분해야 합니다. [확인 설정과 근거](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/gt_settings_followup/evidence.json), [해당 실행 로그](/home/dongjae/.ros/log/python_2628721_1792563151189.log).

실측 소스 설정만 보고 현재 실측 노드가 그 설정으로 실행 중이라고 판단하지 않았습니다. 공유 출력 경로의 프로세스 설정 파일은 검토 중 새 시뮬레이션 설정으로 교체됐습니다. 이번 보고서는 복사한 설정과 확인 시각을 [증거 파일](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/evidence.json)에 구분해 기록합니다.

확인 당시 설치 패키지는 `gelslam.egg-link`로 `build/gelslam`을 참조했습니다. 확인한 8개 핵심 모듈의 build 경로는 소스와 같은 파일로 연결됐습니다. 현재 설치본이 과거 복사본이라는 전제로 수정 항목을 만들지 않았습니다.

근거: [실측 소스 설정](/home/dongjae/gelslam/src/gelslam/config/real_config.yaml:17), [시뮬레이션 소스 설정](/home/dongjae/gelslam/src/gelslam/config/bag_simulation_config.yaml:17), [고정한 설정 복사본](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/config_snapshots/latest_saved_process_config_1422.yaml), [현재 백엔드 구분](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_pose_graph.py:24).

## 2. 가장 먼저 참고할 구현: 원본 IMU 구간 검사

전달 코드의 `preintegrate`는 다음 순서를 사용합니다.

1. 추정 결정을 내리는 시각까지 **수신된** 샘플만 선택합니다.
2. 상태 시작 시각과 목표 시각을 둘러싸는 원본 샘플이 있는지 확인합니다.
3. 양 끝을 둘러싸는 샘플을 포함해 **원본 샘플 간격**을 검사합니다.
4. 검사가 통과한 구간에서 끝점을 보간하고 실제 시간 간격으로 적분합니다.
5. 그래프에 넣기 전에 적분 시간과 두 상태 시각의 차이가 같은지 검사합니다.

우리 `ImuBuffer.samples_covering`은 먼저 끝점을 보간합니다. 접촉 경로의 후속 간격 검사는 이렇게 잘린 구간을 검사할 수 있습니다. 따라서 원본 데이터의 긴 공백이 짧은 영상 구간 안에서 숨겨질 수 있습니다.

같은 제한 `max_sample_gap_s=0.05`로 합성 입력을 확인했습니다.

| 확인 항목 | 결과 |
| --- | --- |
| 원본 IMU 샘플 간격 | 1.0 s |
| 검사 대상 영상 구간 | 1.00 → 1.04 s |
| 우리 경로 | 보간된 40 ms 구간을 연속 구간으로 수락 |
| 전달 경로 | 보간 전에 원본 1.0 s 공백을 거절 |
| 별도 비접촉 예측 검사 | 원본 200 ms 공백을 넣어도 상태 시각 1.2 s까지 예측을 수락 |

이는 설정값 차이가 아니라 **검사 순서와 경로 차이**입니다. 시뮬레이션 bag의 원본 IMU는 모두 5 ms 간격이었습니다. 따라서 이 합성 검사만으로 앞선 시뮬레이션 오차를 설명하지는 않습니다.

권장 수정은 다음과 같습니다.

- 버퍼에서 원본 구간과 양쪽 경계 샘플을 검사하는 공통 함수를 만듭니다. 보간 후의 샘플 간격만으로 원본 입력을 승인하지 않습니다.
- 접촉 최적화, 비접촉 예측, IMU 복구, 백엔드 전달 구간에 같은 정책을 적용합니다. 정지 초기화에 이미 있는 입력 검사도 공통화 대상입니다.
- `_recover_current_frame_with_imu`의 마지막 측정 유지 방식은 정책을 명시합니다. 목표 시각의 다음 IMU 샘플을 기다리거나, 제한된 예측으로 구분하고 사용한 마지막 측정 시각과 유지 시간을 기록합니다.
- 원본 공백을 넘은 예측을 유효한 측정 구간으로 표시하지 않습니다. 적분 시간과 상태 시각의 일치도 검사합니다.

완료 기준은 원본 경계 공백, 내부 공백, 샘플 부족, 중복·역행, 늦은 수신을 각 경로에서 검사하는 것입니다. 정상 5 ms 입력의 결과도 유지해야 합니다. 전달 코드의 약 7 ms 제한은 그대로 복사하지 않고 실제 센서 주기와 허용 손실 정책으로 정해야 합니다.

근거: [전달 구현의 원본 검사](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/snapshot/src/ptm_core/tactile_inertial/continuous_graph.py:56), [적분 시간 검사](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/snapshot/src/ptm_core/tactile_inertial/continuous_graph.py:143), [우리 끝점 보간](/home/dongjae/gelslam/src/gelslam/gelslam/core/inertial.py:206), [구간 생성](/home/dongjae/gelslam/src/gelslam/gelslam/core/inertial.py:256), [비접촉 예측](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:2101), [복구 예측](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:1115), [합성 검사 결과](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/probes.json).

## 3. 우리 시뮬레이션에서 먼저 수정할 부분: 접촉 판정

**이 항목은 전달 구현에서 가져올 완성 기능이 아닙니다.** 전달 PDF 10쪽도 접촉 측정 신뢰도를 후속 과제로 제안합니다. 우리 쪽에는 이미 별도의 재생 근거가 있습니다.

현재 NormalFlow `Frame`은 SDK 마스크를 침식한 뒤 같은 마스크의 픽셀 수로 접촉을 판정합니다. 정합에 쓸 안전한 영역이 작다는 사실과 실제 접촉이 없다는 판단이 결합돼 있습니다.

이전 동일 입력 진단에서 초기 상태를 고정하고 접촉 존재 판정만 침식 전 SDK 마스크로 바꿨습니다. 정합용 침식과 normal/height 잔차는 유지했습니다.

| 공통 조건의 결과 | 기존 판정 | 진단 판정 |
| --- | ---: | ---: |
| 공통 시각 1,358개의 위치 RMSE | 7.223 mm | 2.964 mm |
| GT 접촉 중 비접촉 모드 영상 | 53개 | 13개 |
| IMU 복구 키프레임 | 4개 | 1개 |

이 수치는 **이전 프론트엔드 전용 재생**의 결과입니다. 이후 키프레임 전달과 루프 검색 소스가 변경됐습니다. 이번 검토에서는 그 변경 뒤 같은 bag을 다시 재생하지 않았습니다. 현재 NormalFlow의 침식·접촉 판정 결합은 다시 확인했습니다.

권장 수정은 접촉 존재, 정합 가능 여부, 추정 상태의 유효성을 각각 관리하는 것입니다. 접촉 존재 판정에는 침식 전 마스크 등의 신호를 사용하고, 정합에는 침식한 마스크와 공유점·잔차 검사를 계속 사용합니다. 실제 비접촉 데이터에서도 오검출과 재접촉 성공률을 검사해야 합니다.

완료 기준은 같은 초기 상태와 공통 평가 시각으로 위 재생을 다시 비교하는 것입니다. 연속 접촉 bag 외에 실제 lift가 있는 bag을 포함해야 합니다. 침식을 전부 제거하거나 픽셀 기준만 낮추는 변경을 완료안으로 보지는 않습니다.

근거: [시뮬레이션 상세 보고서](/home/dongjae/gelslam/docs/gelslam_simulation_review_20261021.md), [입력·초기 상태 일치 검증](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/verification.json), [공통 시각 비교](/home/dongjae/gelslam/diagnostics/simulation_review_20261021/analysis.json), [현재 NormalFlow 접촉 판정](/home/dongjae/miniconda3/envs/gel-slam/lib/python3.10/site-packages/normalflow/utils.py:10).

## 4. 실측 초기화: 기존 정지 초기화를 먼저 연결

전달 구현에서 참고할 핵심은 추정값을 계산했다는 사실과 운용 가능한 초기화라는 판단을 구분하는 점입니다. `qualified=false`이면 일반 실행을 거절하고, 진단 실행은 진단 상태로 남깁니다.

우리 소스에는 정지 검출 초기화가 이미 있습니다. 정지 구간의 평균 gyro, 중력 정렬, 정지 앵커의 속도 0을 사용합니다. 그 뒤 첫 접촉 목표 시각까지 IMU를 전파합니다. **정지 앵커의 속도 0을 첫 접촉 상태의 속도 0으로 다시 설정하는 구조가 아닙니다.**

다만 현재 `real_config.yaml`은 정지 초기화를 끄고 초기 중력에 `ground_truth_initial_pose`를 지정합니다. GT 없는 실측 운용을 목표로 한다면 다음 순서가 적절합니다.

1. 기존 `stationary_detection`을 실제로 지지된 정지 구간에서 검증합니다. 이번 메모리 내 설정 검사에서는 활성화가 허용됐고 GT 초기화 옵션이 정지 초기화로 대체됐습니다. 실제 실측 시작 성공을 시험한 것은 아닙니다.
2. 초기화 자격, 실패 사유, 정지 앵커 시각, 첫 접촉 목표 시각을 기록합니다. 정지 상태의 `v=0` 제약은 확인한 정지 시각에만 둡니다.
3. 단일 정지 자세로 분리할 수 없는 중력 방향과 가속도계 bias는 여러 축의 접촉 운동으로 추가 추정합니다. 전달 코드의 조건 검사와 중력 방향 탐색은 이 후속 단계의 참고가 됩니다.

현재 정지 초기화도 가속도계 bias 세 축을 모두 독립적으로 알아내지는 않습니다. 설정 bias에서 시작하고 정지 specific force 크기에 대한 방향 성분을 보정합니다. 한 정지 자세만으로 전체 bias를 식별했다고 보고하면 안 됩니다.

전달 구현의 동적 초기화에도 한계가 있습니다. 중력 방향 2자유도와 가속도계 bias 3축에 대한 단순화된 조건 검사를 사용합니다. PDF 7쪽에서 Rock sliding만 임시 gate를 통과했고, Rock hop·Maple recontact·Wood_handle free는 실패했습니다. 따라서 이 코드를 옮기면 실측 초기화가 해결된다고 판단할 수 없습니다.

`rank=5`, 특이값 비율 0.01, 접촉 이력 길이는 검증할 설계값입니다. 저장된 연결 구간의 끝을 이용한 초기화 이력 선택도 오프라인 정책입니다. 실시간 시작에서는 그 시점에 관측할 수 있는 사건으로 대체해야 합니다. 새 5초 rest 수집 gate 역시 저장 추정기에 정지 초기화가 적용됐다는 뜻은 아닙니다.

완료 기준은 GT 입력을 제거한 독립 실측에서 시작 성공률, 초기화 소요 시간, 첫 접촉의 속도·자세 오차, 실패 사유를 보고하는 것입니다.

근거: [전달 초기화 자격 처리](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/snapshot/src/ptm_core/tactile_inertial/continuous_graph.py:104), [조건 검사](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/snapshot/scripts/continuous_graph_extended_20261020/initialization/joint.py:44), [자격 판정](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/snapshot/scripts/continuous_graph_extended_20261020/initialization/joint.py:174), [우리 초기화 설정 우선순위](/home/dongjae/gelslam/src/gelslam/gelslam/core/static_imu_initialization.py:121), [정지 앵커와 접촉 시각 전파](/home/dongjae/gelslam/src/gelslam/gelslam/core/static_imu_initialization.py:280).

## 5. 주변화 prior: 정보행렬뿐 아니라 우변도 보존

이 항목은 전달 PDF 10쪽의 권고이며, 전달 `continuous_graph.py`에 있는 주변화 코드를 복사하는 작업은 아닙니다. 전달 그래프는 전체 이력을 유지합니다.

우리 `schur_complement`는 축소 정보행렬과 축소 우변을 모두 계산합니다. 그러나 일부 저장 경로가 정보행렬만 넘겨서 우변을 0으로 만듭니다.

| 현재 경로 | 확인한 동작 | 적용 범위 |
| --- | --- | --- |
| `gtsam_vi._refine_current`의 다음 prior | `NavigationPrior.from_information`으로 우변 0 저장 | 현재 프론트엔드 주변화 설정에 해당 |
| `gtsam_pose_graph._marginalize_before` | 축소 우변 계산 후 정보행렬만 prior에 저장 | 로컬 VI와 백엔드 주변화를 켰을 때 해당. 현재 real/simulation 기본 설정은 꺼짐 |
| `gtsam_vi.propagate_prior` | `reduced.rows()`로 행렬과 우변 보존 | IMU 예측 prior. 위 두 경로와 구분해야 함 |

선형화된 정규방정식이 `H δ = g`라면, 주변화 뒤에도 `H_s δ = g_s`를 유지해야 합니다. `H_s`만 저장하면 선형화 기준 상태에서 보정량을 0으로 유도하는 다른 prior가 됩니다. 정확히 수렴해 `g_s`가 거의 0이면 차이가 작을 수 있습니다. 반복 횟수 제한이나 남은 잔차가 있으면 차이가 생길 수 있습니다.

대수 예제에서 정확한 축소 문제의 해는 약 1이었고, 우변을 0으로 저장한 해는 0이었습니다. 이는 정보 손실의 예제입니다. 실제 궤적 오차가 1이라는 뜻은 아닙니다. 현재 궤적 오차 중 이 정책이 차지하는 비율은 아직 측정하지 않았습니다.

권장 수정은 최종 선형화 기준 상태와 함께 `A, rhs`를 보존하는 것입니다. 이미 적용한 최적화 증분을 두 번 반영하지 않도록 기준 상태의 좌표 규약도 함께 검증해야 합니다. 단순히 기존 중간 우변 하나를 추가하는 변경으로 끝내면 안 됩니다.

완료 기준은 우변이 0이 아닌 문제에서 전체 그래프 해와 주변화 해가 일치하는지 확인하는 것입니다. 고정 기준 상태, 좌표 변환, 재선형화, bias 고정 모드도 포함해야 합니다. 현재 테스트에는 우변을 0으로 저장하는 정책을 확인하는 항목이 있습니다. 그 테스트의 통과는 정확한 Schur prior와 같다는 증거가 아닙니다.

근거: [축소 정보행렬과 우변 계산](/home/dongjae/gelslam/src/gelslam/gelslam/core/marginal_prior.py:60), [프론트엔드 prior 저장](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_vi.py:1148), [백엔드 prior 저장](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_pose_graph.py:330), [우변을 보존하는 IMU 예측 prior](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_vi.py:503), [현재 정책 테스트](/home/dongjae/gelslam/src/gelslam/test/test_marginal_prior.py:150).

## 6. 함께 참고할 진단과 후속 개선

### 상태와 측정의 출처를 기록

전달 코드는 접촉 factor의 실제 기준·목표 상태, 측정 ID, 이용 가능 시각, 중복 상태 쌍의 거절 사유를 기록합니다. 초기화에서 사용한 측정을 다시 그래프에 넣을 때는 초기화 posterior를 추가 독립 prior로 중복 투입하지 않습니다.

우리도 다음 항목을 함께 기록하면 복구와 루프 문제를 찾기 쉽습니다.

- 마지막 수락 상태 `t_a`, IMU 예측 목표 `t_b`, 실제 예측 상태 시각, 발행 시각과 상태 경과 시간.
- 접촉 측정의 실제 기준 영상과 목표 영상, 접촉 존재, 공유점 수, 수락·거절 사유.
- 예측만 한 상태와 접촉 최적화를 수락한 상태, 초기화 자격.
- 새 factor의 최적화 전후 비용과 factor 종류별 비용. 초기화와 주변화의 정보 재사용 여부.

전달 코드의 같은 상태 쌍 중복 방지는 유용합니다. 다만 여러 측정이 같은 영상을 공유해서 생기는 상관관계까지 모두 제거하는 기능은 아닙니다.

현재 우리 코드에는 `pose_timestamp`, `pose_valid`, 영상의 IMU 대기 처리가 이미 있습니다. PDF의 과거 coasting timestamp 지적을 그대로 미해결 항목으로 넣지 않았습니다. 새 진단은 이 필드를 실제 상태와 일치시키는 회귀 검사에 연결하면 됩니다.

### 로컬 VI를 켤 때 배치 처리의 일관성 검사

전달 그래프는 새 상태를 순서대로 추가하고 갱신합니다. 우리 `LocalMapping.process`는 새 키프레임 배치를 모두 추가한 뒤 한 번 풉니다. 로컬 창이 10개이면 마지막 창을 기준으로 최적화합니다.

따라서 큰 배치를 한 번에 넣은 경우와 작은 배치로 나눠 넣은 경우의 로컬 VI 결과를 비교해야 합니다. 필요하면 상태 추가와 창 이동을 순차 처리하거나 배치 크기를 제한합니다. 이는 최근 수정한 루프 검색의 배치 누락과 다른 검토 항목입니다. 현재 real/simulation 기본 설정은 로컬 VI가 꺼져 있으므로 해당 설정의 직접 원인으로 해석하지 않습니다.

근거: [배치 추가 후 최적화](/home/dongjae/gelslam/src/gelslam/gelslam/core/local_mapping.py:43), [로컬 창 선택](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_pose_graph.py:457).

### 장치 시각과 noise는 센서에 맞게 검증

`recorded_source_clock._unwrap_device_ms`는 `2^32 ms` rollover를 가정합니다. 앞선 합성 검사에서 `[59990, 59995, 0, 5]`는 거절됐습니다. 약 60,000 ms 주기의 장치 시각을 사용할 때는 해당 장치 규약에 맞는 unwrap이 필요합니다. 현재 확인한 설정은 기본 `header` 경로이므로 이 조건부 문제를 현재 실행 전체의 시각 오류로 확대하지 않습니다.

Noise 값도 그대로 복사하면 안 됩니다. 전달 코드의 가속도계·자이로 density는 0.08, 0.01입니다. 우리 실측 소스 설정은 0.002, 0.00016이고 별도 `imu_residual_weight=0.001`도 사용합니다. 기본 density 숫자만 비교하면 실제 factor 가중치를 잘못 해석할 수 있습니다.

전달 PDF도 noise를 Allan 분석으로 검증한 값이 아니라 설계 가중치라고 설명합니다. 정지·온도·진동 조건을 구분한 센서 데이터와 독립 운동에서 잔차·공분산을 검증한 뒤 조정해야 합니다.

근거: [장치 시각 처리](/home/dongjae/gelslam/src/gelslam/gelslam/core/recorded_source_clock.py:53), [이전 입력 검사](/home/dongjae/gelslam/docs/gelslam_handoff_audit_20261021.md), [전달 noise 설정](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/snapshot/src/ptm_core/tactile_inertial/continuous_graph.py:15), [현재 실측 설정](/home/dongjae/gelslam/src/gelslam/config/real_config.yaml:17).

## 7. 현재 유지할 부분과 그대로 옮기지 않을 부분

| 항목 | 판단 |
| --- | --- |
| IMU 원점 상태와 lever arm | 우리 코드에도 구현돼 있음. 유지하고 좌표 변환 회귀 검사에 사용 |
| 실제 시간 간격과 native IMU factor | 이미 사용함. 원본 입력 검사를 보강 |
| bias random walk의 `sqrt(Δt)` | 이미 사용함. 재접촉 때 bias·속도를 무조건 초기화하지 않음 |
| direct normal/height 잔차 | 유지할 근거가 있음. 전달 Pose3 방식과의 우열은 동일 입력 실험이 필요 |
| IMU 복구 키프레임 | `tactile_valid=false`와 예측 prior 유지가 이미 있음. 가짜 접촉 측정으로 바꾸지 않음 |
| 키프레임 전달·루프 검색 | 현재 소스에 수정 반영을 확인함. 다시 미수정 결함으로 보고하지 않음 |
| 전체 iSAM2 전환 | 계산 시간, 메모리, 지연, 동일 데이터 정확도 비교 후 결정 |
| 전달 Pose3 factor의 직접 추가 | 우리 IMU 융합 프론트엔드 pose를 독립 접촉 측정처럼 추가하면 IMU 정보가 중복될 수 있음 |
| extrinsic 숫자 복사 | 우리 설정은 `T_GI`, 전달 그래프 입력은 `T_IG`. 서로 역변환 관계이므로 방향부터 확인 |
| GTSAM 설정 복사 | 요구 binding과 적분 구현이 다름. 같은 입력·중력·bias의 예측과 Jacobian을 먼저 비교 |
| 초기화 gate와 noise 숫자 복사 | 센서와 운동에 대한 검증이 필요. 통과율이나 정확도를 보장하지 않음 |

근거: [IMU/Gel 원점과 속도 변환](/home/dongjae/gelslam/src/gelslam/gelslam/core/imu_navigation.py:1), [현재 접촉 factor](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_pose_graph.py:146), [IMU 복구](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:1115), [키프레임 전달 수정](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:1025), [현재 루프 검색](/home/dongjae/gelslam/src/gelslam/gelslam/core/local_mapping.py:161).

## 8. 변경 파일과 검증

이번 작업은 보고서와 진단 기록만 추가했습니다. 기존 제품 소스와 운용 설정의 변경은 그대로 보존했습니다.

| 자료 | 내용 |
| --- | --- |
| 이 보고서 | 적용할 변경, 현재 유지할 기능, 완료 기준 |
| [evidence.json](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/evidence.json) | PDF·동봉 소스 검증, 소스/build 해시, 설치 경로, 설정 구분, 검사 범위 |
| [probes.json](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/probes.json) | 같은 제한의 원본 IMU 공백 비교, 비접촉 공백, Schur 대수 예제, 초기화 설정 검사 |
| [targeted_tests.log](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/targeted_tests.log) | 관련 테스트 **117개 통과**, 3.85초 |
| [설정 복사본](/home/dongjae/gelslam/diagnostics/implementation_comparison_20261021/config_snapshots/latest_saved_process_config_1422.yaml) | 검토 시각의 저장 프로세스 설정. 소스·설치·이전 재생 설정도 같은 폴더에 고정 |

테스트 범위는 주변화 prior, 정지 초기화, 영상/IMU 동기, midpoint 적분, 키프레임 전달 회귀입니다. 테스트 후 8개 핵심 모듈이 바뀌지 않았는지 다시 확인했습니다. 동봉 bundle의 34개 파일과 source manifest의 25개 파일은 해시 검사에 통과했습니다. Python 파일의 구문도 검사했습니다.

이번에는 새 전체 bag 재생이나 전달 iSAM2 그래프의 수치 실행을 수행하지 않았습니다. 전달 구현과 우리 구현의 정확도 우열을 측정한 보고서는 아닙니다. 원본 간격 비교는 입력 거절 로직을 확인한 검사입니다. 이전 시뮬레이션 결과는 당시 소스·공급 설정·로그와 함께 보존한 별도 근거로 사용했습니다.

수정 후에는 **발행 당시 추정 결과와 최종 루프 보정 결과**, **접촉으로 지지된 구간과 전체 실행 구간**을 구분해 평가해야 합니다. 위치·회전 오차 외에 유효 상태 비율, 재접촉 직전 예측 오차, 재접촉 최적화 후 오차, 실패 사유, 처리 지연을 함께 보고하는 것이 적절합니다.
