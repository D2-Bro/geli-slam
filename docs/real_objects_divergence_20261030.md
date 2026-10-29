# real-objects 데이터의 프론트엔드 발산: 원인, bag 문제, 수정 (2026-10-30)

대상:
- real: `/home/dongjae/dataset/GeLISLAM/real-objects-12-spatial-rebased-raw-time-20261029/trial_001`–`012`
- sim: `/home/dongjae/dataset/GeLISLAM/fixture-*-gelslam-image` (8개)

브랜치는 `claude/divergence-analysis`입니다. 분석 도구는 `scripts/real_objects_diagnostics/`에 있습니다.
실행 결과는 `output/real_batch_20261030/`에 있습니다 (git 제외).
real 설정은 `src/gelslam/config/real_objects_config.yaml`(새 파일)이고, sim 설정은 `bag_simulation_config.yaml`입니다.

## 1. 결과 요약

### real (실시간 재생, 원본 bag, 프론트엔드)

- 기준: 기존 `real_far_config` 계열 설정과 `recorded_rest_event` 초기화 (`output/real_batch_20261030/E0_raw`)
- 수정 후: `real_objects_config.yaml` (`FINAL_live`)
- 실행: `bag_replay_launch.py backend:=false` + `ros2 bag play --clock --rate 1.0`. 다른 CPU 작업은 돌리지 않았습니다.

| trial | 기준: 위치 RMSE / 회전 RMSE / 최종 | 수정 후: 위치 RMSE / 회전 RMSE / 최종 | GT 무효 비율 |
|---|---|---|---|
| 001 | 91.1 mm / 1.58° / 106.2 mm | 6.0 mm / 0.64° / 5.3 mm | 0.5% |
| 002 | 152.7 mm / 1.52° / 186.3 mm | 6.9 mm / 0.82° / 9.1 mm | 2.8% |
| 003 | 117.4 mm / 1.89° / 195.7 mm | 18.6 mm / 1.04° / 23.9 mm | 10.4% |
| 004 | 120.2 mm / 2.79° / 170.2 mm | 10.2 mm / 1.31° / 16.7 mm | 5.5% |
| 005 | 227.9 mm / 2.05° / 232.9 mm | 17.7 mm / 0.92° / 24.3 mm | 14.6% |
| 006 | 초기화 실패 (상태 없음) | 14.9 mm / 1.46° / 25.2 mm | 4.4% |
| 007 | 82.7 mm / 1.95° / 113.2 mm | 5.6 mm / 1.28° / 10.2 mm | 5.2% |
| 008 | 70.4 mm / 2.45° / 88.0 mm | 7.6 mm / 1.85° / 14.8 mm | 4.9% |
| 009 | 140.4 mm / 1.65° / 120.9 mm | 3.9 mm / 0.71° / 3.6 mm | 2.3% |
| 010 | 초기화 실패 (상태 없음) | 13.0 mm / 1.27° / 20.5 mm | 17.9% |
| 011 | 97.4 mm / 1.78° / 140.4 mm | 4.2 mm / 1.12° / 5.3 mm | 1.1% |
| 012 | 201.9 mm / 1.19° / 222.9 mm | 7.7 mm / 0.74° / 12.4 mm | 1.4% |
| 평균 | 130.2 mm / 1.88° / 157.6 mm (10개) | 9.7 mm / 1.10° / 14.3 mm (12개) | |

- 수정 후의 실시간 결과는 오프라인 결정적 재생과 trial마다 거의 같습니다 (평균 9.7 mm 대 9.7 mm, 상태 수 차이 0–3개).
- trial_001의 GT 경로 길이는 66 s 동안 2.83 m입니다. 최종 오차 5.3 mm는 그 0.2%입니다.

### sim (실시간 재생, 8개 fixture)

| sequence | 기준: 백엔드 평가 / 프론트엔드 평가 | 수정 후: 백엔드 평가 / 프론트엔드 평가 |
|---|---|---|
| rock_line | 1.30 mm / 1.3 mm | 1.20 mm / 1.2 mm |
| rock_zigzag_gap | 3.73 mm / 4.0 mm | 2.83 mm / 3.4 mm |
| avocado_arc | 0.74 mm / 0.7 mm | 0.51 mm / 0.5 mm |
| walnut_loop_gap | 6.23 mm / 6.2 mm | 4.14 mm / 4.1 mm |
| rock_long_sweep | 0.92 mm / 1.0 mm | 0.91 mm / 1.0 mm |
| rock_long_figure8 | 0.81 mm / 0.9 mm | 0.83 mm / 0.9 mm |
| plane_yaw | 접촉 미검출 | 접촉 미검출 |
| periodic_return_gap | 접촉 미검출 | 접촉 미검출 |

- 기준과 수정 후는 같은 코드입니다. 차이는 `contact_velocity_sigma_mps: 0.3` 하나입니다 (`output/sim_batch_20261029/runs_base_20261030`, `runs_final_20261030`).
- 백엔드 평가는 파이프라인 자체의 `trajectory_metrics.json`(위치 RMSE)입니다. 프론트엔드 평가는 이 문서의 방법입니다.


- 평가: 프론트엔드 IMU 원점 상태와 GT `/ground_truth/imu_pose`를 비교합니다.
  - 시간: source 기준입니다. 카메라는 PTS, IMU는 device counter입니다. GT는 trial마다 추정한 mocap 지연(3.5–10.5 ms)을 뺐습니다.
  - 정렬: trial 그룹마다 첫 상태에서 SE(3) 정렬을 한 번만 합니다. 곡선 맞춤은 하지 않습니다.
  - GT 무효 구간(2.3절)은 평가에서 뺍니다.
- 기준 실행에서 trial_006과 trial_010은 초기화에 실패해서 상태가 없었습니다.

## 2. bag 분석

### 2.1 이미지 시간 (가장 큰 bag 문제)

- `/gelsight/image_raw`의 header는 host 수신 wall 시간입니다 (`/gelsight/timing`의 `host_wall_ns`와 같음).
- `source_pts_ns`는 host monotonic 시계의 값입니다. 수신 시간(`host_monotonic_ns`)과의 차이:
  - 12개 trial 모두 p1 52 ms, 중앙값 62.5 ms, p99 75–84 ms입니다.
  - drift는 100 s당 ±1.4 ms 이내입니다.
- 결과: header 모드에서는 이미지가 IMU보다 약 61 ms 늦게 놓입니다. 그리고 프레임마다 ±10 ms의 jitter가 있습니다.
  - 기존 설정은 `image_to_imu_time_offset_s: 0`이었습니다.
  - 회전이 1–2 rad/s일 때 61 ms는 4–7°의 촉각–IMU 불일치에 해당합니다.

### 2.2 IMU 시간

- `/imu/data_raw`의 header도 수신 시간입니다. 샘플이 묶음으로 도착합니다.
  - trial마다 372–1075개(2–5%)의 샘플이 이전 샘플과 같거나 더 이른 header를 갖습니다. tracker는 이 샘플을 버립니다.
  - header 간격은 최대 57 ms입니다. `max_sample_gap_s` 50 ms보다 큽니다.
- `/imu/device_time_ms`는 60 000 ms마다 0으로 돌아가는 counter입니다. wrap을 풀면 간격이 5 ms로 균일합니다.
  - 실제 결측은 trial_003, 004, 011에 15 ms 간격이 1–2개 있을 뿐입니다.
  - 수신 시간은 device 시간의 하한선보다 평균 1.6 ms, p99 8 ms 늦습니다.
- 결과: IMU를 강하게 믿는 설정(3.3절)에서는 이 jitter와 결측이 큰 오차를 만듭니다.
  - 같은 G 설정에서 원본 header를 쓰면 위치 RMSE 평균이 115 mm입니다. source 시간을 쓰면 9.7 mm입니다.

### 2.3 mocap GT

- Motive frame 번호에 빈 곳이 있습니다. 빠진 frame의 비율은 trial_001 0.4%, trial_010 28.1%, trial_005 16.1%이고, 나머지는 1–8%입니다.
- 재획득 직후에 물리적으로 불가능한 자세 점프가 있습니다.
  - 예: trial_010에서 한 샘플(8 ms)에 11.3°, trial_005에서 최대 64.8°입니다. gyro의 최대값은 2.3 rad/s입니다.
  - 이 점프는 `tracking_valid`나 `mean_error_m`로 표시되지 않습니다.
- IMU로 검사한 결과(0.1 s 창에서 GT 상대 회전과 gyro 회전의 차이가 1°를 넘는 샘플):
  - GT 무효 비율은 0.5%(trial_001) – 17.9%(trial_010)입니다.
  - 무효 구간을 빼지 않으면 trial_010과 trial_005의 RMSE에 GT 오류가 섞입니다.
- mocap 수신 시간은 source IMU 시간보다 3.5–10.5 ms 늦습니다 (gyro와 mocap 각속도의 정렬로 추정).

### 2.4 첫 접촉 전 동작

- GO(`trial_ready`) 뒤 1–4 s에 손이 움직이기 시작합니다. 첫 접촉은 zero-velocity 앵커보다 3.1–7.4 s 늦습니다.
- 이 시간 동안 IMU만 적분하므로, 첫 접촉 velocity seed의 오차가 0.13–0.29 m/s입니다.
  - GT 속도는 0.000–0.048 m/s입니다.
  - 수직 σ는 0.03 m/s이고, z-score는 2–3입니다.
- `max_propagation_s: 5`에서는 trial_006(6.25 s)과 trial_010(7.42 s)이 초기화에 실패해서, trial 전체를 추적하지 못했습니다.

## 3. 파이프라인 원인과 수정

### 3.1 촉각 정합의 초기값 (주원인 1)

- 증거 (trial_001, 연속 접촉 프레임 1157쌍, `tactile_probe.py`):
  - NormalFlow 단독 정합의 CCS는 중앙값 0.997이고, 100% 통과합니다. GT와의 차이는 0.12 mm, 0.24°입니다. 그래서 영상과 normal 모델은 정상입니다.
  - VI 결합 해는 1464개 중 697개가 CCS 검사에서 거부되었습니다 (기존 설정).
- 기구: 연속 프레임 사이에 gel 표면은 약 0.15 mm 움직이지만, 회전은 약 0.9°입니다. IMU는 gel에서 0.139 m 떨어져 있어서, IMU 원점 속도는 약 40 mm/s입니다. VI velocity 오차(30–80 mm/s) 때문에 IMU 예측 pose가 프레임당 1–2 mm(약 25 pixel) 어긋나고, dense normal 해가 잘못된 최소점으로 갑니다.
- 수정: `frontend_initialize_from_imu_preintegration: false`. 결합 해는 이전 상태에서 시작하고, IMU factor는 그대로 씁니다.
  - trial_001: 거부 685 → 0, 위치 RMSE 68.7 → 15.7 mm
  - sim에서는 영향이 거의 없습니다 (평균 2.4 → 2.2 mm).
  - velocity가 좋아진 G 설정에서도 이 값은 필요합니다. `true`로 되돌리면 9.7 → 102.6 mm입니다.

### 3.2 IMU 적분 공분산 (주원인 2)

- `gtsam_integration_sigma_m: 0.01`은 코드 기본값(1e-4)의 100배입니다. 이 값 때문에 위치와 velocity의 결합이 약해서, velocity가 관측되지 않습니다.
- VI velocity 오차(접촉 중, 12개 trial):

  | integration sigma | 중앙값 | p90 |
  |---|---|---|
  | 0.01 | 24.4 mm/s | 78.1 mm/s |
  | 0.001 | 6.1 mm/s | 19.1 mm/s |
  | 0.0001 | 3.8 mm/s | 15.6 mm/s |

- velocity 오차는 비접촉 구간(lift-off 후 IMU coast)에서 위치 오차가 됩니다. 기존 설정에서는 구간 97개의 중앙값이 16.6 mm였고, 최대 325 mm였습니다.
- 수정: real 설정만 0.001로 바꿉니다.
  - sim에서는 0.001이 긴 sequence를 나쁘게 합니다 (sweep 1.0 → 36 mm, 촉각 거부 0 → 145). 그래서 sim 설정은 0.01을 유지합니다. 이 차이의 원인은 확인하지 않았습니다 (4절).
- 중간 해결책으로 `coast_velocity_fit.model: contact_kinematic`도 구현했습니다. lift-off velocity = gel 접촉 velocity + ω × (R r)입니다.
  - integration sigma가 0.01일 때는 구간 오차의 합계가 2763 → 970 mm(오프라인 적분)로 줄었습니다.
  - 0.001에서는 VI velocity가 더 정확합니다. 그래서 최종 설정에서는 끕니다.

### 3.3 첫 접촉 velocity와 gyro bias

- `static_initialization.contact_velocity_sigma_mps: 0.3` (새 옵션): 첫 접촉 prior의 velocity 공분산에 0.3² m²/s²를 더합니다.
  - 그러면 첫 촉각 프레임들이 velocity를 다시 추정합니다. loosely-coupled 초기화(촉각 위치와 IMU 사전적분으로 v0를 최소제곱으로 푸는 방법)를 결합 해 안에서 푸는 것과 같습니다.
  - 0.25 s 시점의 velocity 오차: 약 190 → 58–93 mm/s
  - 첫 0.25 s의 위치 오차: 9–22 mm(8개 trial) → 0.4–10.6 mm
  - 위치 RMSE 평균: 16.6 → 10.4 mm
  - sim 평균: 2.4 → 1.9 mm
- `gyroscope_bias_random_walk: 2.2e-6` (기존 2.2e-5, real 설정만):
  - 기존 값에서는 추정 gyro bias의 z 성분이 70 s 동안 2 mrad/s 움직였고, yaw 오차가 4–8°가 되었습니다.
  - 정지 평균으로 보면 bias는 안정적입니다.
  - 회전 RMSE 평균: 2.27° → 1.10°
- `gtsam_vi_prior_gyro_bias_sigma_rps`는 이 경로에서 결과를 바꾸지 않습니다.

### 3.4 시간 처리 (real 설정)

- `timestamp_source: recorded_source`: bag 경로 없이 `/gelsight/timing`과 `/imu/device_time_ms`를 받아서, warmup 뒤 offset을 고정합니다. 이렇게 하면 수신 jitter와 중복 IMU 샘플이 없어집니다.
- `image_to_imu_time_offset_s: -0.0615`: source 시계도 평균 수신 지연을 그대로 가지므로 필요합니다.
  - 오프셋 시험(4개 trial): 회전 RMSE는 0 ms에서 1.33–2.42°, −61.5 ms에서 0.61–1.89°입니다. −40 ms와 −80 ms는 그 사이 값입니다.
- source 시간으로 다시 쓴 bag 복사본: `output/real_batch_20261030/bags_srcclock/` (`rewrite_source_time.py`). 이 bag에는 `timestamp_source: header`와 오프셋 0을 씁니다.

### 3.5 초기화와 실시간 처리

- `max_propagation_s: 15` (기존 5): trial_006과 trial_010을 추적할 수 있게 됩니다.
- `/geli/events` 구독을 별도 콜백 그룹으로 옮겼습니다 (executor 스레드 2 → 3).
  - 이미지 그룹에 있을 때는 CPU 부하에서 이벤트가 20–60 s 늦게 도착했습니다. 그 사이 IMU 버퍼(30 s)가 정지 구간 시작을 지워서, 초기화에 실패했습니다 (trial_007, 012). 이것은 제가 이전에 넣은 버그입니다.
- 실시간 실행은 CPU 부하가 있으면 이미지를 버립니다 (queue 깊이 1). 이 손실은 `replaced_images`에 집계되지 않습니다. 부하가 없을 때의 실시간 결과는 오프라인 결정적 재생과 같습니다 (상태 수 차이 1개 이내).

## 4. 남은 문제

- 접촉 중 drift: 위치 약 1 mm/s, 회전은 trial당 수 °입니다. 비접촉 구간의 IMU coast 오차가 남은 위치 오차의 대부분입니다.
- 백엔드 (`backend:=true`, feedback 꺼짐, 최종 설정, 실시간 12개 trial, `FINAL_backend`):
  - 파이프라인 자체 평가(GelSight pose 위치 RMSE)에서 raw odometry와 비교한 결과입니다.
  - 좋아진 trial: 001 5.9 → 4.2, 002 7.9 → 5.3, 003 19.4 → 2.8, 004 10.8 → 8.3, 009 4.4 → 2.3, 010 13.0 → 3.9 mm
  - 나빠진 trial: 005 19.2 → 53.9, 006 15.2 → 33.0, 011 3.9 → 8.4, 012 7.6 → 16.1 mm
  - 12개 trial 모두 종료 제한 안에 끝났습니다. 수정 전에는 키프레임이 350개를 넘어서 끝나지 못한 경우가 있었습니다.
  - trial_005에서는 Global PGO가 4번 실행되었습니다. 마지막(키프레임 164개)의 비용은 74 550 → 1866이었습니다. 로그에 loop 수용 정보가 없어서, 잘못된 loop인지는 확인하지 못했습니다.
- sim:
  - plane_yaw와 periodic_return_gap은 접촉을 검출하지 않습니다. 이전 분석과 같은 원인입니다 (평면에서 SDK 접촉 마스크가 0 pixel).
  - sim에서 integration sigma를 줄이면 긴 sequence에서 촉각 거부가 늘어납니다. 원인은 확인하지 않았습니다.
- GT: mocap frame 결측과 점프 때문에, 프레임 단위 정밀도는 CCS 검사에 쓸 정도가 아닙니다 (GT pose에서 CCS 통과율 71%).

## 5. 실행 방법

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch gelslam bag_replay_launch.py \
  config_path:=$(ros2 pkg prefix gelslam)/share/gelslam/config/real_objects_config.yaml \
  backend:=false backend_feedback:=false use_sim_time:=true
# 다른 터미널: tracker 시작 후 원본 bag을 재생합니다.
ros2 bag play /home/dongjae/dataset/GeLISLAM/real-objects-12-spatial-rebased-raw-time-20261029/trial_001/bag --clock
```

일괄 실행과 평가: `scripts/real_objects_diagnostics/README.md`를 참조하세요.
