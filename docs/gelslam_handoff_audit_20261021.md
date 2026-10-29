# GelSLAM 시스템 점검 보고서

**점검일 2026년 10월 21일 · 실행 기록 12시 58분 KST · 상태 수정과 검증 필요**

GT 없이 실측 데이터를 처리하려면 **초기화, 입력 시각, IMU 공백 검사**를 먼저 보완해야 합니다. 로컬 VI의 일괄 처리와 주변화는 별도 검증이 필요합니다.

이 보고서는 2026년 10월 21일 점검 기록을 정리한 문서입니다. 실행 결과는 당시 마지막 저장 실행에 해당합니다. 제품 소스와 설정은 변경하지 않았습니다.

## 시뮬레이션 추가 점검

**연속 접촉 시뮬레이션에서 접촉을 놓치는 현상을 현재 소스로 재현했습니다.** GT 접촉 영상 1,392개 중 53개가 비접촉 모드로 들어갔습니다. 그중 41개는 SDK 마스크에 NormalFlow 침식을 추가한 뒤 픽셀 수가 300개 미만으로 줄어든 경우입니다.

초기 상태와 평가 정렬을 같게 유지하고 접촉 존재 판정만 침식 전 마스크로 바꾼 진단에서는, 공통 시각 1,358개의 위치 RMSE가 **7.22 mm에서 2.96 mm로 감소**했습니다. IMU 복구 키프레임도 4개에서 1개로 줄었습니다. 정합용 마스크와 제품 소스는 유지했습니다.

따라서 프론트엔드에서는 접촉 존재 판정과 정합용 마스크 침식의 분리를 먼저 검증할 필요가 있습니다. 이 데이터의 추정기 비접촉 모드는 실제 lift를 뜻하지 않습니다.

자세한 조건, 시간별 오차, 마스크 영상은 [시뮬레이션 실행 점검 보고서](/home/dongjae/gelslam/docs/gelslam_simulation_review_20261021.md)에 정리했습니다.

## 핵심 결과

| 순서 | 항목 | 확인한 사실 | 적용 범위 |
| --- | --- | --- | --- |
| 1 | GT 없는 초기화 | GT 초기 자세가 없으면 초기화를 기다립니다. | 실측 프론트엔드 |
| 2 | 입력 시각 | IMU 헤더 시각 중복 쌍 90개를 확인했습니다. | 최근 입력 bag 전체 |
| 3 | IMU 공백 검사 | 50 ms 제한을 넘는 공백이 통과했습니다. | 접촉 구간 검사와 비접촉 예측 재현 |
| 4 | 루프 검색 | 여러 키프레임을 한 번에 받으면 마지막 목표만 검색합니다. | 백엔드 루프 검색 재현 |
| 5 | 로컬 VI 일괄 처리 | 일부 관측이 최적화 전에 윈도 밖으로 나갑니다. | 로컬 VI 활성화 시 |
| 6 | 주변화 prior | 다음 상태로 넘기는 prior에서 우변을 버립니다. | 프론트엔드 및 조건부 로컬 백엔드 |

> **원인 판단 범위**
>
> 최근 입력 bag에는 50 ms 초과 IMU 공백이 없었습니다. 최근 실행에서는 로컬 VI도 꺼져 있었습니다. 따라서 위 재현 결과를 최근 위치 오차의 원인으로 바로 연결할 수는 없습니다.

## 최근 저장 실행

### 실행 구성

| 항목 | 실행에 공급된 값 |
| --- | --- |
| 센서 설정 | 소스의 `real_config.yaml` |
| 프론트엔드 | IMU와 normal/height 잔차를 함께 최적화 |
| IMU 적분 | `midpoint` |
| IMU 목적함수 가중치 | `0.001` |
| 중력 초기화 | GT 초기 Gel 자세 사용 |
| GT 속도 초기화 | 꺼짐 |
| GT 이전 상태 진단 | 꺼짐 |
| 시간 경로 | IMU 헤더 시각, 추가 offset `0 s` |
| 로컬 VI | 꺼짐 |
| 백엔드 | 프론트엔드 초기값을 쓰는 4DoF yaw/XYZ 루프 PGO |
| 백엔드 피드백 | 켜짐 |

### 실행 결과

| 지표 | 값 | 해석 |
| --- | ---: | --- |
| 종료 평가의 대응 자세 | 52개 | 저장 실행의 평가 대상 |
| 위치 RMSE | **61.77 mm** | 원래 값 `0.061766092 m` |
| 회전 RMSE | **0.877°** | 원래 값 `0.877208193°` |
| 프론트엔드 refine 호출 | 98회 | 입력 영상 전체 수와 다름 |
| VI 수락 | 48회 | refine 호출 기준 |
| VI 품질 거절 | 50회 | refine 호출 기준 |
| 백엔드 키프레임 | 30개 | 저장된 키프레임 수 |
| 수락 루프 | 0개 | 이 실행에서 루프 보정 효과는 평가할 수 없음 |

평가는 첫 대응 자세에서 정한 하나의 고정 SE(3) 변환을 사용했습니다. 전체 궤적을 맞추는 정렬이나 scale fitting은 사용하지 않았습니다.

실행은 Ctrl-C로 중단됐습니다. 이 결과는 전체 bag 성능이나 GT 없는 운용 성능을 뜻하지 않습니다. 종료 과정에서 `visualizer_node`가 `exit -11`로 끝났습니다. 추적 오차와의 인과관계는 확인되지 않았습니다.

근거: [프론트엔드 실행 요약](/home/dongjae/gelslam/gelslam_online/tracker_process_summary.json), [백엔드 실행 요약](/home/dongjae/gelslam/gelslam_online/backend_process_summary.json), [종료 궤적 평가](/home/dongjae/gelslam/gelslam_online/trajectory_evaluation/trial004_VIO/trajectory_metrics.json), [ROS 종료 로그](/home/dongjae/.ros/log/2026-10-21-12-58-35-539382-rvi-B460MDS3HV2-2558426/launch.log)

## 우선 조치와 검증 기준

### GT 없는 실측 초기화

실측 설정은 `static_initialization.enabled=false`이고, `frontend_gravity_initialization=ground_truth_initial_pose`입니다. GT 초기 자세가 없으면 해당 trial의 접촉 초기화를 기다립니다. GT를 제공하지 않은 최소 재현에서 초기화 완료 횟수는 **0회**였습니다.

정지 탐지 초기화는 이미 구현돼 있습니다. 실측 전용 설정에서 이를 켜고, 실제 정지 구간과 재시작 조건을 확인해야 합니다. 정지 한 자세만으로 가속도 bias 전체와 중력을 독립적으로 식별할 수는 없습니다.

**검증 기준:** GT 입력을 제거한 실행에서 실제 정지 근거로 초기화를 완료하고, 재시작 조건도 확인합니다.

시뮬레이션 설정의 `1.0~2.9 s` 정지 구간은 해당 시뮬레이션 전용입니다. 이 설정에서는 정지 초기화가 YAML의 GT 중력·속도 옵션보다 우선합니다.

근거: [실측 초기화 설정](/home/dongjae/gelslam/src/gelslam/config/real_config.yaml:23), [GT 자세 대기 경로](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:790), [초기화 우선순위](/home/dongjae/gelslam/src/gelslam/gelslam/core/static_imu_initialization.py:121)

### 실제 입력 시각

최근 실행이 읽은 bag의 전체 IMU 입력을 확인했습니다.

| 항목 | IMU 헤더 시각 | IMU 장치 시각 |
| --- | ---: | ---: |
| 샘플 수 | 6,344개 | 6,344개 |
| 같은 시각의 인접 쌍 | **90개** | **0개** |
| 역행한 인접 쌍 | 0개 | 0개 |
| 간격 중앙값 | 약 5.03 ms | 5 ms |
| 간격 95백분위수 | 약 8.81 ms | 5 ms |
| 최대 간격 | 약 16.54 ms | 5 ms |

장치 시각은 `12,215~43,930 ms`에서 모두 5 ms 간격으로 증가했습니다. 반면 최근 실행은 헤더 시각을 사용했습니다. 현재 버퍼는 중복 헤더 시각을 거절하며, 최근 실행 로그에도 IMU 거절이 기록돼 있습니다.

**검증 기준:** 센서 측정 시각과 수신 시각을 구분하고, 하나의 시간 보정 정책을 선택합니다. 선택한 시간축의 중복·역행·간격과 offset 적용 횟수를 확인합니다.

`recorded_source` 경로에는 별도 문제가 있습니다. 현재 코드는 `2^32 ms` rollover만 처리합니다. `[59990, 59995, 0, 5]` 입력은 예외로 거절했습니다. 60,000 ms 주기의 장치 데이터에 이 경로를 쓸 때 수정이 필요합니다. 최근 입력은 rollover를 지나지 않았고, 실행도 헤더 경로를 사용했습니다.

전달 자료의 `-38/-32/-34 ms` 값은 별도 clock mapping 후 저장된 입력에 적용한 값입니다. 현재 원시 헤더에 바로 적용해서는 안 됩니다.

근거: [입력 bag 검사](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/input_bag_audit.json), [장치 시각 rollover 처리](/home/dongjae/gelslam/src/gelslam/gelslam/core/recorded_source_clock.py:53), [IMU 입력 버퍼](/home/dongjae/gelslam/src/gelslam/gelslam/core/inertial.py:184), [재생 입력 로그](/home/dongjae/.ros/log/python3_2558556_1792555119157.log)

### IMU 원본 공백 검사

| 재현 경로 | 원본 샘플 공백 | 설정 제한 | 실제 결과 |
| --- | ---: | ---: | --- |
| 접촉 프론트엔드 구간 검사 | 1.0 s | 0.05 s | 영상 구간 끝점을 보간한 뒤 0.04 s 간격으로 보여 통과 |
| 비접촉 예측 | 0.2 s | 0.05 s | 목표 시각 1.2 s까지 예측하고 재시작을 요청하지 않음 |

**조치:** 보간 전에 원본 샘플 간격을 검사합니다. 접촉 프론트엔드, 비접촉 전파, 백엔드에 같은 검사 규칙을 적용합니다.

**검증 기준:** 원본 간격이 허용값을 넘으면 보간된 끝점 간격이 짧아도 구간을 거절합니다. 허용값 자체는 센서 데이터로 정합니다. 전달 구현의 7 ms 값을 그대로 복사하는 것으로 해결되지는 않습니다.

근거: [끝점 보간](/home/dongjae/gelslam/src/gelslam/gelslam/core/inertial.py:206), [구간 샘플 구성](/home/dongjae/gelslam/src/gelslam/gelslam/core/inertial.py:256), [비접촉 전파](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:2139), [최소 재현 코드](/home/dongjae/gelslam/diagnostics/module_review_20261021/reproduce.py)

### 여러 키프레임의 일괄 처리

| 경로 | 재현 조건 | 결과 | 조치 |
| --- | --- | --- | --- |
| 루프 검색 | K0~K4를 한 번에 입력 | K4만 검색하여 K1~K3의 검색 기회 누락 | 검색 완료 위치를 기록하고 각 새 목표를 처리 |
| 로컬 VI | K0~K5를 한 번에 입력, 윈도 크기 3 | K1의 Z 오차 2 mm 유지 | 관측을 최적화한 후 윈도 이동 |

같은 관측을 순서대로 로컬 VI에 넣으면 K1의 오차는 0.001 mm 미만이었습니다. 이 로컬 VI 오류는 해당 기능을 켤 때 적용됩니다. 최근 실행에서는 로컬 VI가 꺼져 있었습니다.

**검증 기준:** 순차 입력과 일괄 입력이 같은 관측을 최적화하고, 각 새 목표 키프레임에 같은 루프 검색 기회를 제공해야 합니다.

근거: [로컬 VI 일괄 처리](/home/dongjae/gelslam/src/gelslam/gelslam/core/local_mapping.py:89), [루프 검색](/home/dongjae/gelslam/src/gelslam/gelslam/core/local_mapping.py:167), [로컬 윈도 선택](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_pose_graph.py:462)

### 주변화 prior의 우변 보존

주변화는 이전 상태를 제거하고 그 정보를 남은 상태의 prior로 전달합니다. 현재 두 경로는 다음과 같습니다.

| 경로 | 최적화 증분 계산 | 다음 prior에 저장하는 값 | 최근 실행 적용 |
| --- | --- | --- | --- |
| 프론트엔드 | 우변 사용 | 정보행렬 보존, 우변 0 | 적용됨 |
| 로컬 백엔드 | Schur 축소 계산 | 축소 정보행렬 보존, 우변 제거 | 적용 안 됨 |

로컬 백엔드 경로는 `local_vi_enabled=true`와 `marginalization_enabled=true`를 함께 설정할 때 사용합니다. 프론트엔드의 0 우변 저장 정책은 현재 테스트도 의도적으로 검사합니다.

축소 우변이 0이 아니면 정확한 Schur prior와 다릅니다. 점검의 간단한 수치 예제에서는 정확한 축소 prior의 해가 `y=1`, 우변을 버린 해가 `y=0`이었습니다. 이 결과는 근사의 차이를 보여줍니다. 최근 실측 오차의 원인까지 입증하지는 않습니다.

**검증 기준:** 같은 초기값과 관측을 사용하여 우변을 보존한 prior, 현재 prior, 전체 배치 최적화 결과를 비교합니다.

근거: [프론트엔드 prior 저장](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_vi.py:1148), [로컬 백엔드 주변화](/home/dongjae/gelslam/src/gelslam/gelslam/core/gtsam_pose_graph.py:355), [prior 구성](/home/dongjae/gelslam/src/gelslam/gelslam/core/marginal_prior.py:114), [현재 정책을 검사하는 테스트](/home/dongjae/gelslam/src/gelslam/test/test_marginal_prior.py:150)

## 실행 환경과 설정 일치 여부

### Python 환경

| 항목 | 확인한 값 |
| --- | --- |
| 실행 Python | `/home/dongjae/miniconda3/envs/gel-slam/bin/python` |
| Python | 3.10.12 |
| NumPy / SciPy | 1.26.4 / 1.15.2 |
| GTSAM | 4.3.0 |
| GTSAM API | ComponentWise, midpoint, ORB-SLAM3 호출 성공 |
| 관련 모듈 | ROS, cv_bridge, NormalFlow, GelSight SDK import 성공 |
| 기본 `python3` | Conda base의 3.13.11로, 실행 환경과 다름 |

### 소스와 설치 사본

현재 `PYTHONPATH`의 첫 경로는 `build/gelslam`입니다. 그 아래 `gelslam`은 현재 소스 디렉터리로 연결됩니다. 실제 import에서도 이미지·IMU 대기 API와 `ImuBuffer.latest_timestamp`를 확인했습니다. 최근 실행 로그에도 대기 동작이 기록돼 있습니다.

| 항목 | 현재 소스와 실제 import 경로 | install 사본 |
| --- | --- | --- |
| 이미지·IMU 대기 기능 | 최신 기능 있음 | `tracker_node.py`, `inertial.py`가 소스와 다름 |
| 실측 IMU 목적함수 가중치 | `0.001` | `1.0` |

설치 사본의 차이를 최근 실행의 기능 누락으로 해석해서는 안 됩니다. 반대로 설치 사본만 읽는 다른 환경에서는 동작이 달라질 수 있습니다. 가중치 차이는 목적함수의 차이이며, 위치 오차가 1,000배라는 뜻은 아닙니다.

기본 `config.yaml`의 `gelsight_T_imu`는 `null`입니다. `bag_replay_launch.py`는 설정 인자를 생략하면 설치된 `config.yaml`을 선택합니다. 백엔드는 보정된 외부 변환을 요구하므로 실측 실행에는 검증한 설정 경로를 명시해야 합니다. 최근 실행은 소스의 `real_config.yaml`을 명시했습니다.

근거: [환경과 사본 해시](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/evidence.json), [프론트엔드 공급 설정](/home/dongjae/gelslam/gelslam_online/tracker_process_config_supplied.yaml), [백엔드 공급 설정](/home/dongjae/gelslam/gelslam_online/backend_process_config_supplied.yaml)

## 전달 구현과 현재 구현의 차이

| 항목 | 전달 구현 | 현재 시스템 |
| --- | --- | --- |
| 내부 상태 원점 | IMU | IMU |
| 접촉 측정 | NormalFlow의 상대 pose | normal/height 직접 잔차 |
| 접촉 프론트엔드 | 저장된 NormalFlow 결과 사용 | IMU와 접촉 잔차를 함께 최적화 |
| 비접촉 경로 | 연속 graph의 IMU 상태 체인 유지 | 별도 IMU 예측 경로 |
| 백엔드 | iSAM2 | 로컬 VI와 4DoF 루프 PGO 분리 |
| 최근 로컬 VI 상태 | 해당 없음 | 꺼짐 |
| 중력 초기화 | 초기 구간에서 방향과 상태를 함께 추정 | 최근 실측 실행은 GT 초기 자세 사용 |
| 외부 변환 표기 | `T_IG` | `gelsight_T_imu = T_GI` |

`T_IG`와 `T_GI`는 서로 역변환입니다. 값을 그대로 복사하면 안 됩니다. 양쪽 모두 IMU 상태에서 Gel pose를 구할 때 전체 외부 변환을 사용합니다.

전달 자료가 비교한 소스는 커밋 `3791b3634555`입니다. 현재 작업 트리는 그 이후 변경을 포함합니다. 두 구현의 성능 우열은 같은 데이터로 비교하지 않았습니다.

### 이전 지적 중 보완된 항목

| 항목 | 현재 상태 | 남은 확인 사항 |
| --- | --- | --- |
| 비접촉 pose 시각 | `pose_timestamp`와 `pose_valid`로 실제 상태 시각 전달 | 마지막 수신 IMU 시각과 영상 목표 시각 구분 |
| 종료 궤적 평가 | `pose_valid=false`인 상태 제외 | 유효 상태 수와 제외 사유 함께 보고 |
| 루프 보정 | 4DoF 보정을 로컬 IMU 상태·중력과 분리 | 이전 6DoF 경로의 우려를 현재 경로에서 재평가 |

영상 목표 시각을 `t_j`, 예측 상태 시각을 `t_s`라고 할 때, 두 시각이 다르면 `pose_valid=false`입니다. 현재 소스는 이전 예측의 시각을 영상 시각으로 단순히 덮는다는 과거 지적을 보완했습니다.

근거: [IMU와 Gel 좌표 변환](/home/dongjae/gelslam/src/gelslam/gelslam/core/imu_navigation.py:12), [비접촉 상태 발행](/home/dongjae/gelslam/src/gelslam/gelslam/core/tracker.py:2184), [평가 대상 선택](/home/dongjae/gelslam/src/gelslam/gelslam/nodes/pose_graph_node.py:253), [4DoF 루프 보정](/home/dongjae/gelslam/src/gelslam/gelslam/core/loop_pose_graph.py:96)

## 검증 결과

| 검사 | 결과 | 범위 |
| --- | --- | --- |
| 전달 패키지 무결성 | 34개 파일의 크기·SHA256 일치 | manifest 대상 |
| 전달 snapshot 무결성 | 25개 파일 일치 | snapshot 대상 |
| Python 문법 | 15개 파일 통과 | 전달 snapshot |
| 현재 기능 테스트 | **750개 통과 · 9개 실패 · 12개 준비 단계 오류** | 18.40초 |
| ROS 관련 테스트 | 35개 통과 | 쓰기 가능한 로그 경로 지정 후 |
| 전달 수치 테스트 | 버전 검사에서 중단 | GTSAM 4.3a1 요구, 현재 4.3.0 |

기능 테스트에서는 저작권·flake8·pep257 검사를 제외했습니다. 처음 발생한 ROS 로그 쓰기 제한은 점검 디렉터리를 지정한 뒤 해소했습니다. 위 전체 기능 테스트 수치는 해당 조건으로 다시 실행한 최종 결과입니다.

남은 실패에는 오래된 테스트 대역의 필드 누락과 현재 API의 불일치가 있습니다. 누락 항목에는 `normalflow`, `imu_config`, `_projection_cache`, `target_z_height_correction`이 포함됩니다. 소스 문자열을 찾는 진단 코드는 `StopIteration`으로 실패했습니다. 비접촉 bias 테스트 일부는 현재 사용하지 않는 `inertial_state` 경로를 기대합니다.

**9개 실패와 12개 준비 단계 오류를 제품 오류 21건으로 계산해서는 안 됩니다.** 제품 동작 오류는 앞의 최소 재현 결과와 구분해야 합니다.

전달 수치 테스트의 버전 검사는 우회하지 않았습니다. 기존 GTSAM도 교체하지 않았습니다.

## 권장 작업 순서

1. **실행 경로 고정** — Python 환경, import 경로, 공급 설정을 기록하고 소스·설치 사본 차이를 정리합니다.
2. **GT 없는 초기화와 시간 보정** — 실제 정지 근거로 시작하고, 하나의 시간축과 offset 정책을 검증합니다.
3. **IMU 공백 검사 수정** — 보간 전 원본 샘플 간격을 모든 사용 경로에서 검사합니다.
4. **키프레임 일괄 처리 수정** — 관측 최적화와 각 목표의 루프 검색 기회를 보존합니다.
5. **주변화 비교** — 우변 보존 여부를 전체 배치 최적화 결과와 비교합니다.
6. **짧은 실제 hop 평가** — 재접촉 갱신 전 예측 오차와 갱신 후 오차를 따로 측정합니다. 온라인 당시 결과와 종료 후 보정 결과도 구분합니다.

## 근거 자료

| 자료 | 용도 |
| --- | --- |
| [환경과 재현 결과](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/evidence.json) | 환경, 사본 해시, 설정, 최소 재현, 테스트 요약 |
| [입력 bag 검사](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/input_bag_audit.json) | 헤더 시각과 장치 시각 비교 |
| [최종 기능 테스트 로그](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/pytest_writable_log.log) | 통과·실패·준비 단계 오류 상세 |
| [ROS 테스트 로그](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/ros_tests_writable_log.log) | 관련 35개 테스트 결과 |
| [전달 테스트 실행 로그](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/bundle_core_checks.log) | GTSAM 버전 검사 중단 근거 |
| [테스트 실행 명령](/home/dongjae/gelslam/diagnostics/handoff_audit_20261021/test_command.txt) | 같은 조건의 테스트 실행 |
| [전달 구현 설명 PDF](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/GelI-SLAM-Implementation-KO-20261021.pdf) | 전달 구현의 구조와 비교 기준 |
| [전달 입력 규약](/home/dongjae/Downloads/GelI-SLAM-Tactile-IMU-20261021/INPUT_CONTRACT_KO.md) | 좌표계, 단위, 시간축, replay 입력 조건 |

점검 당시 호스트에는 GelSLAM 노드, ros2 bag, RViz 프로세스가 없었습니다. 저장 로그와 입력 bag을 읽기 전용으로 확인했으며, 새 bag 재생이나 전달 구현으로의 교체는 수행하지 않았습니다.
