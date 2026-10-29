<h1 align="center">
    GelSLAM: A Real-Time, High-Fidelity, and Robust <br/> 3D Tactile SLAM System
</h1>

<div align="center">
  <a href="https://joehjhuang.github.io/" target="_blank">Hung-Jui Huang</a> &nbsp;•&nbsp;
  <a href="https://www.aminmirzaee.com/" target="_blank">Mohammad Amin Mirzaee</a> &nbsp;•&nbsp;
  <a href="https://www.cs.cmu.edu/~kaess/" target="_blank">Michael Kaess</a> &nbsp;•&nbsp;
  <a href="https://siebelschool.illinois.edu/about/people/all-faculty/yuanwz" target="_blank">Wenzhen Yuan</a>
</div>

<h4 align="center">
  <a href="https://joehjhuang.github.io/gelslam"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Web.svg" alt="Website" width="10px"/> <b>Website</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2508.15990"><img src="assets/arxiv.png" alt="arXiv" width="28px"/> <b>Paper</b></a> &nbsp;&nbsp;&nbsp; &nbsp;
  🤗 <a href="https://huggingface.co/datasets/joehjhuang/GelSLAM_dataset"> <b>Dataset</b></a>
</h4>


<div align="center">
<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp; 
<a href="https://rpl.ri.cmu.edu/" target="_blank">
  <img height="20" src="assets/rpl.png" alt="RPL">
</a>
</div>

<p align="center">
  <img src="assets/teaser.gif" alt="GelSLAM">
</p>

GelSLAM is a real-time 3D SLAM system that enables ultra-high-fidelity object shape reconstruction and precise object pose tracking relying on tactile sensing alone. It accomplishes what was previously out of reach for tactile-only systems. For more details and results, please visit our [website](https://joehjhuang.github.io/gelslam) and the [arXiv paper](https://arxiv.org/abs/2508.15990). 

---

### 💻 실행 환경

현재 촉각·IMU 프런트엔드와 15차원 백엔드는 ROS 2 bag 입력을 사용합니다.
아래 환경은 이 작업 공간에서 확인한 환경입니다.

- Ubuntu 22.04, ROS 2 Humble Desktop, Python 3.10.
- Conda 환경 `gel-slam`. 빌드와 실행에 같은 Python을 사용합니다.
- NumPy 1.26.4, OpenCV 4.11.0.86, SciPy 1.15.2, Open3D 0.18.0.
- GTSAM 4.3.0 Python 바인딩. `ImuFactorErrorMode.ComponentWise`와
  `PreintegrationParams.setImuFactorErrorMode`가 필요합니다.
  IMU 적분에는 아래 패치의 midpoint API를 사용합니다.
- RViz 또는 Open3D 화면을 사용하려면 데스크톱 화면 연결이 필요합니다.

### 🛠️ 설치

#### 1. ROS 2와 시스템 패키지

먼저 [ROS 2 Humble 공식 설치 절차](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)에
따라 Ubuntu용 ROS 패키지 저장소를 등록합니다. 그다음 설치합니다.

```bash
sudo apt update
sudo apt install ros-humble-desktop ros-humble-cv-bridge \
  ros-humble-rosbag2-storage-default-plugins \
  python3-colcon-common-extensions python3-rosdep \
  build-essential cmake git libboost-all-dev libtbb-dev ffmpeg wget unzip
```

#### 2. 저장소와 Python 환경

아래 명령은 새 설치 기준입니다. 기존 저장소 또는 `gel-slam` 환경이 있으면
해당 생성 단계를 건너뜁니다. Conda가 설치되어 있어야 합니다.

```bash
git clone https://github.com/D2-Bro/geli-slam.git ~/gelslam
cd ~/gelslam
conda create -n gel-slam python=3.10 pip -y
conda activate gel-slam
python -m pip install 'setuptools<80' wheel \
  'numpy==1.26.4' 'opencv-python==4.11.0.86' -r requirements.txt
python -m pip install 'empy==3.3.4' catkin_pkg lark \
  colcon-common-extensions matplotlib pytest
```

`requirements.txt`는 NormalFlow와 GelSight SDK를 Git 저장소에서 설치합니다.
ROS 2 Humble의 `cv_bridge`와 함께 사용하도록 위 명령에서 NumPy 1.26.4를 지정합니다.
GTSAM은 다음 단계에서 별도로 설치합니다.

#### 3. GTSAM Python 바인딩

버전 번호만 확인하지 말고 필요한 API도 확인해야 합니다.
아래 소스 리비전은 현재 작업 공간에 설치된 GTSAM 리비전입니다.
아래 명령은 GelSLAM 저장소 루트에서 실행합니다. 중간 자세 및 ORB-SLAM3 IMU 적분을 위해 두 GTSAM 패치를 순서대로 적용합니다.
빌드 절차는 [GTSAM Python 설치 문서](https://github.com/borglab/gtsam/blob/71a25ca36c084cbad1f872e812d6d97fbadfdb05/python/README.md)를 기준으로 합니다.

```bash
conda activate gel-slam
mkdir -p ~/libraries
git clone https://github.com/borglab/gtsam.git ~/libraries/gtsam-4.3.0
git -C ~/libraries/gtsam-4.3.0 checkout 71a25ca36c084cbad1f872e812d6d97fbadfdb05
git -C ~/libraries/gtsam-4.3.0 apply "$PWD/patches/gtsam-midpoint-integration.patch"
git -C ~/libraries/gtsam-4.3.0 apply "$PWD/patches/gtsam-orb-slam3-integration.patch"
python -m pip install -r ~/libraries/gtsam-4.3.0/python/dev_requirements.txt
cmake -S ~/libraries/gtsam-4.3.0 -B ~/libraries/gtsam-4.3.0/build-conda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DGTSAM_PYTHON_VERSION=3.10 -DGTSAM_BUILD_PYTHON=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF
cmake --build ~/libraries/gtsam-4.3.0/build-conda -j2
cmake --install ~/libraries/gtsam-4.3.0/build-conda
cmake --build ~/libraries/gtsam-4.3.0/build-conda --target python-install
python - <<'PYTHON'
import gtsam
p = gtsam.PreintegrationParams([0.0, 0.0, -9.80665])
p.setImuFactorErrorMode(gtsam.ImuFactorErrorMode.ComponentWise)
p.setUseMidpointIntegration(True)
p.setUseOrbSlam3Integration(True)
assert p.getUseOrbSlam3Integration() and not p.getUseMidpointIntegration()
print('GTSAM API OK:', gtsam.__file__)
PYTHON
```

#### 4. ROS 패키지 빌드와 확인

`imu_config.integration_method`는 `midpoint`만 허용합니다. 이 방식은
IMU 구간 중간 자세로 평균 가속도를 회전합니다. 양끝 가속도·각속도를 평균하고
회전을 합성합니다.
이 설정은 초기 IMU 전파, 프론트엔드 VI, 비접촉 전파, 복구, 백엔드 IMU factor에
함께 적용됩니다. 설정을 바꾼 뒤 노드를 다시 실행합니다.

```bash
cd ~/gelslam
conda activate gel-slam
source /opt/ros/humble/setup.bash
python -m colcon build --symlink-install --packages-select gelslam_msgs gelslam \
  --cmake-args -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
               -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
source install/setup.bash
python - <<'PYTHON'
import rclpy, gtsam, normalflow, gs_sdk
from cv_bridge import CvBridge
from gelslam_msgs.msg import KeyFrameMsg
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
assert 'navigation_valid' in KeyFrameMsg.get_fields_and_field_types()
print('GelSLAM imports OK')
PYTHON
ros2 launch gelslam bag_replay_launch.py --show-args
```

`KeyFrameMsg`가 변경되었으므로 두 패키지를 함께 빌드합니다.
다른 Python으로 만든 기존 빌드가 있으면 `build/`, `install/`을 정리한 뒤 다시 빌드합니다.
소스의 설정 파일을 변경해도 설치된 설정 사본은 다를 수 있습니다.
아래 실행 예제는 `config_path`에 소스 설정의 절대 경로를 직접 전달합니다.

### ⚡ ROS 2 bag 실행

#### 1. 데이터와 설정 준비

bag, 모델, 결과 파일은 Git에 포함하지 않습니다. 데이터를 별도로 준비합니다.
`ros2 bag info /absolute/path/to/bag`로 토픽과 메시지 형식을 확인합니다.

| 설정 파일 | 이미지 / IMU 토픽 | 적용 범위 |
|---|---|---|
| `src/gelslam/config/bag_simulation_config.yaml` | `/image` / `/imu/data_raw` | 제공된 시뮬레이션 bag. 고정 초기화 구간은 figure8 데이터 전용 |
| `src/gelslam/config/real_config.yaml` | `/gelsight/image_raw` / `/imu/data_raw` | 기록 데이터용 설정. 현재 값은 특정 데이터에 맞춰져 있음 |
| `src/gelslam/config/config.yaml` | `image` / `/imu/data` | 센서 설정의 시작점. `gelsight_T_imu`는 `null` |

선택한 YAML에서 다음 값을 데이터에 맞게 수정합니다.

- `device_config.calibration_model_path`: 해당 GelPad의 `nnmodel.pth` 경로.
- `device_config.torch_num_threads`: 표면 복원의 Torch CPU 연산 스레드 수. 기본값은 `4`이며, 모델 초기화 전에 적용됩니다. 같은 프로세스의 Torch CPU 연산에 적용됩니다.
- `device_config.image_topic`, `imu_config.topic`, `imu_config.frame_id`: bag의 실제 값.
- `imu_config.gelsight_T_imu`: IMU 좌표를 GelSight 좌표로 변환하는 보정 행렬.
- `imu_config.image_to_imu_time_offset_s`: 이미지와 IMU의 측정 시각 차이.

시뮬레이션 설정은 `static_initialization.enabled: true`이며,
IMU 헤더 시각 1.0~2.9초를 정지 초기화 구간으로 사용합니다.
전체 bag을 처음부터 재생해야 이 구간을 받을 수 있습니다.
다른 데이터에서는 실제 정지 구간에 맞게 수정하거나 초기화 방식을 바꿉니다.
현재 정지 초기화는 설정에 남아 있는 GT 기반 중력·속도 초기화보다 우선합니다.

두 기록 데이터 설정에는 GT 토픽을 사용하는 옵션이 남아 있습니다.
GT가 없는 데이터에서는 `frontend_gravity_initialization`과
`gravity_initialization`을 `fixed_world_z`로 설정할 수 있지만,
이 방식은 초기 촉각 좌표의 Z축과 중력 방향이 맞을 때만 유효합니다.
`frontend_first_trial_velocity_from_gt`는 `false`로 설정하고,
`trajectory_evaluation.enabled`와 `trajectory_evaluation.state_velocity.enabled`도
사용하지 않으면 `false`로 설정합니다.
정지 초기화와 시간 보정의 상세 조건은 [IMU 준비 문서](docs/imu_preparation.md)를 확인합니다.

속도 평가를 켜면 `trajectory_evaluation.state_velocity.truth_path`를 해당 데이터의
`trajectory_truth.npz`로 변경합니다.
저장소에 있는 `/home/dongjae/...` 경로는 각 설치 환경의 경로로 변경해야 합니다.

#### 2. 터미널 A: GelSLAM 시작

각 터미널에서 같은 환경과 작업 공간을 불러옵니다.
아래 예제는 설정을 수정한 시뮬레이션 bag을 사용합니다.

```bash
conda activate gel-slam
source /opt/ros/humble/setup.bash
cd ~/gelslam
source install/setup.bash
ros2 launch gelslam bag_replay_launch.py \
  config_path:="$PWD/src/gelslam/config/bag_simulation_config.yaml" \
  data_dir:="$PWD/data/rock_figure8" sequence_name:=rock_figure8 \
  backend:=true rendering:=false rviz:=true \
  publish_coverage:=true skip_background_check:=true save_states:=true \
  use_sim_time:=true
```

이 launch 파일은 bag 재생기를 시작하지 않습니다.
노드가 준비된 후 터미널 B에서 bag을 재생합니다.
처음 10개 이미지는 배경으로 사용하므로 접촉이 없는 이미지여야 합니다.
백엔드는 보정된 `gelsight_T_imu`와 시간에 맞는 IMU 측정 구간이 필요합니다.

#### 3. 터미널 B: bag 재생

```bash
conda activate gel-slam
source /opt/ros/humble/setup.bash
cd ~/gelslam
source install/setup.bash
ros2 bag info /absolute/path/to/bag
ros2 bag play /absolute/path/to/bag --clock --rate 1.0
```

`/absolute/path/to/bag`를 `metadata.yaml`이 있는 실제 bag 디렉터리로 바꿉니다.
프런트엔드 처리가 입력 속도를 따라가지 못하면 재생 속도를 낮춥니다.
현재 시뮬레이션 설정의 이미지 큐 깊이는 1이므로 오래된 이미지는 생략될 수 있습니다.
bag 재생이 끝나면 터미널 A에서 `Ctrl+C`를 누르고 결과 저장이 끝날 때까지 기다립니다.

접촉 중 VI 최적화 결과는 CCS, 수치 유효성, IMU 예측 일관성 검사로 수용합니다.
SCR만 기준보다 낮으면 현재 최적화 상태와 prior를 유지하고 현재 영상을 즉시 키프레임으로 만듭니다.
공유점이 10개 미만이거나 CCS가 낮으면 이 경로를 사용하지 않습니다.

연속 접촉 중 CCS, 수치 유효성, IMU 예측 일관성 검사 실패가
`gelslam_config.track_config.contact_failure_keyframe_interval_s` 동안 지속되면
IMU 예측 상태로 현재 접촉 영상을 키프레임으로 만듭니다. 기본값은 0.5초입니다.
첫 품질 검사 실패 영상의 센서 시각부터 계산하며, 생성 후 같은 간격으로 반복합니다.
재접촉 정합 실패 또는 대응점 부족은 이 시간을 기다리지 않습니다. IMU 구간이
유효하면 현재 프레임을 즉시 IMU 예측 상태로 승인하고 새 기준 키프레임으로 만듭니다.
다음 프레임은 이 기준으로 정합합니다. IMU가 유효하지 않으면 마지막 승인 상태를 유지합니다.
접촉 추적 성공, 비접촉, 품질 검사 이외의 정합 실패는 품질 검사 실패 타이머를 초기화합니다.
이 상태는 `imu_only`이고 촉각 성공 시각은 갱신하지 않습니다. 15D prior는 IMU로 전파합니다.
정상 추적의 `keyframe_max_interval_s`는 별도 설정입니다.

무접촉 시간은 따로 계산합니다. `imu_config.max_coast_duration_s`를 넘으면 새 구간을
준비하고 접촉 영상 쌍을 한 번 수용한 뒤 발행합니다. IMU가 연속이면 이동 중에도
재시작합니다. IMU 간격이 `imu_config.max_sample_gap_s`를 넘으면 새로운 정지 구간이
필요합니다. `FrameMsg`와 `FramePoseMsg`의 `tracking_status`, `pose_valid`,
`pose_timestamp`는 추정 상태와 실제 자세 시각을 구분합니다.

`imu_config.coast_velocity_fit.enabled: true`는 최근 접촉 수락 위치에서
비접촉 진입 속도를 다시 계산하는 옵션입니다. 코드 기본값은 `false`이며,
현재 시뮬레이션 재생 설정과 figure8 전용 설정에서는 `true`입니다.
`model: inertial_quadratic`은 해당 접촉 시각까지의 IMU 운동을 제거한 뒤
위치 곡선을 맞춥니다. `model: quadratic`은 접촉 위치에 직접 2차 곡선을 맞춥니다.
두 방법 모두 IMU 원점과 고정 trial 좌표를 사용합니다. 비접촉 중에는 별도 예측에만
보정을 적용하고 마지막 접촉 수락 상태와 prior는 유지합니다. 같은 접촉 기준에서
IMU 복구를 수락하면 복구 시각의 위치에 `delta_v * 경과 시간`을 더하고 속도에
`delta_v`를 더합니다. 전파한 prior의 target 위치·속도 분포도 같은 양만큼 이동합니다.
prior의 전체 행, 우변, 상관정보와 다른 시각의 기준 pose는 유지합니다. 피팅 보정은
확정값으로 취급하며 새 독립 관측 factor를 추가하지 않습니다. 복구 수락 후 보정값을
지우고 보정된 복구 상태에서 다음 IMU 전파를 시작하므로 같은 보정을 중복 적용하지 않습니다.
기본 조건은 이력 0.4초, 표본 5개 이상, 시간 폭 0.3초 이상, 접촉 표본 간격
0.15초 이하, 곡선 적합 오차 0.05 mm 이하, 속도 변화 20 mm/s 이하입니다.
조건을 충족하지 않으면 VI 속도를 유지합니다. 접촉 중단, IMU 복구,
trial 변경, 백엔드 좌표 보정 시에는 접촉 이력을 끊습니다.

`src/gelslam/config/figure8_imu_refined_experiment.yaml`은 제공된 긴 figure8 bag
전용 설정입니다. 1.0–2.9초의 정지 IMU 초기화를 사용합니다. 현재 이 설정과
`bag_simulation_config.yaml`에는 비접촉 진입 속도 보정과 정지 자이로 바이어스
불확실성 계산만 적용합니다. 촉각 가중치는 `1.0`, `gtsam_integration_sigma_m`은
`0.01`로 유지합니다. 다른 물체와 실제 센서에는 별도 검증이 필요합니다.
이 설정은 `static_initialization.gyro_bias_uncertainty_from_samples: true`도 사용합니다.
정지 구간의 마지막 측정 시각에서 자이로 바이어스 평균의 불확실성을 계산합니다.
계산에는 표본 시각, 자이로 잡음 density, 바이어스 random walk를 사용합니다.
구간 이후의 IMU 전파는 이 초기 공분산을 사용합니다. 코드 기본값은 `false`이며,
현재 두 시뮬레이션 설정에서는 `true`입니다.
아래 보고서의 최종 수치는 가중치 조정도 포함한 과거 실험 결과입니다.
현재 1·2번만 적용한 설정의 성능 수치로 사용하면 안 됩니다.
측정값, 실패한 실험, 복구 시점의 제한은
[`docs/imu_coast_experiments_20261021.txt`](docs/imu_coast_experiments_20261021.txt)에 있습니다.

주요 launch 옵션은 다음과 같습니다.

| 옵션 | 기본값 | 기능 |
|---|---|---|
| `config_path` | 설치된 `config.yaml` | 실행 노드가 읽는 YAML 경로 |
| `data_dir` | 현재 디렉터리 | `gelslam_online/` 결과를 만들 상위 디렉터리 |
| `backend` | `true` | 15차원 백엔드 최적화. `false`이면 원래 프런트엔드 궤적을 기록 |
| `rendering` | `true` | Open3D 화면 |
| `rviz` | `true` | RViz와 표시용 노드 |
| `publish_coverage` | `true` | 백엔드 표면 데이터 발행 |
| `save_states` | `false` | 트래커·백엔드 상태 저장 |
| `use_sim_time` | `true` | bag 재생기의 `/clock` 사용 |

화면 없이 실행하려면 `rendering:=false rviz:=false publish_coverage:=false`를 지정합니다.
RViz의 원본 영상은 `/gelslam/rviz/image`를 사용합니다. 표시용 노드가
`device_config.image_topic`의 영상을 이 토픽으로 전달합니다.
`debug_visualization_enabled: false`이면 normal map, 대응점, 현재 접촉면은
발행하지 않습니다. 백엔드의 전역 표면을 보려면 `publish_coverage:=true`가 필요합니다.
백엔드 사용 시 재구성 결과는 `<data_dir>/gelslam_online/reconstructed_mesh.ply`입니다.
궤적 평가를 켜면 `<data_dir>/gelslam_online/trajectory_evaluation/<sequence_name>/`에
평가 결과를 저장합니다. `backend:=false` 실행은 백엔드 재구성 메시를 만들지 않습니다.

### Git 관리 범위

소스, 테스트, 패키지 정의, 설정, 기본 보정 예제, README는 Git에서 관리합니다.
`build/`, `install/`, `log/`, `data/`, `ckpt/`, `diagnostics/`, `output/`,
`gelslam_online/`, `gelslam_offline/`, Python 캐시는 제외합니다.
로컬 진단·실험 도구와 개별 실험 보고서도 제외합니다.
단, 회귀 테스트가 직접 불러오는 `scripts/`의 Python 소스 7개는 함께 관리합니다.
실행에 필요한 외부 데이터와 해당 센서의 모델은 사용자가 별도로 준비합니다.

### 테스트

빌드 후 같은 환경에서 회귀 테스트를 실행할 수 있습니다.
일부 진단 테스트는 Plotly를 추가로 사용합니다.

```bash
python -m pip install plotly
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest src/gelslam/test -q \
  --ignore=src/gelslam/test/test_copyright.py \
  --ignore=src/gelslam/test/test_flake8.py \
  --ignore=src/gelslam/test/test_pep257.py
```

위 명령은 기능 테스트를 실행합니다. ROS의 `launch_testing` 플러그인과
설치된 pytest 버전의 충돌을 피하도록 외부 플러그인 자동 로딩을 끕니다.
저작권·코드 형식 검사는 별도 항목입니다.

### 기존 영상 데모의 적용 범위

아래 Live, Live Replay, Offline 및 논문 재현 명령은 원본 프로젝트의 영상 기반
인터페이스입니다. 현재 15차원 백엔드는 동기화된 IMU·항법 상태가 필요합니다.
영상만 입력하는 예제와 샘플 다운로드는 현재 백엔드의 기본 실행 절차가 아닙니다.
현재 촉각·IMU 경로를 실행하려면 위 ROS 2 bag 절차를 사용합니다.
실제 센서의 Live 모드에는 보정 행렬과 별도의 동기화된 IMU 발행기가 필요합니다.

---

### 📌 Prerequisites
Please calibrate your sensor and prepare a configuration file before running GelSLAM on your own data or sensors.

- **Sensor Calibration**:  
  Calibrate your sensor using the [GelSight SDK](https://github.com/joehjhuang/gs_sdk). **Re-calibration is required if the GelPad is replaced or a different sensor is used.** Calibration typically takes under one hour. An example calibration is provided at `src/gelslam/resources/example_calibration/`.

- **Configuration File**:  
  Copy the appropriate YAML to your data directory. Set `device_config.calibration_model_path` to your calibration model path (`model/nnmodel.pth`). Configure the image/IMU topics, mounting transform, time offset, and initialization for your sensor, as described in the ROS 2 bag procedure above.

---

### 🚀 Running GelSLAM

The standard launches run `tracker_node` and `pose_graph_node` as separate ROS
processes. Keyframes travel on `keyframe`; optimized 15D states return on
`backend_feedback` and are applied at an image boundary. The backend uses the
latest combined GelSight (normal + mean height), IMU and bias factors, with projection reuse and a residual-only
error path. See [ROS process architecture](docs/slam_thread_architecture.md).


Set `gelslam_config.track_config.debug_visualization_enabled: false` to skip
normal RGB images, current-contact point clouds and correspondence images,
including their construction. This applies to both `gelslam_node` and
`tracker_node`. The default is `true`, including when the key is omitted.
Set the value before starting the node. Pose estimation and frame/keyframe
publication continue when it is `false`. This switch is independent of the
Open3D `rendering` launch option.

When debug visualization is enabled, the tracker publishes a normal correspondence image on
`/gelslam/rviz/normal_correspondences` for each processed surface image.
The reference normal map is on the left. The current normal map is on the
right. Solid green lines connect the normal-flow reference pixels to their
projected current-image positions at the final frontend pose.
The image shows timestamps, the solve status, and the number of displayed
pairs. A rejected pose can still show its final projections. Reference
initialization, no contact, and a solve without a final pose show no lines.

The supplied RViz configuration enables **Normal Point Correspondences**.
For another RViz configuration, add an **Image** display and select this topic.
`gelslam_config.track_config.correspondence_max_lines: 30` samples at most
30 pairs at evenly spaced indices in each frame. If fewer pairs are available,
the display shows all available pairs. Set `0` to show all pairs.
This value does not change tracking. The image contains frontend projections; backend corrections do
not change these pairs. The initial ten background images have no normal
correspondence image.

The original project provided the following three reconstruction modes.
The video-only modes do not supply the navigation inputs required by the current backend. Before running GelSLAM, please ensure you have gone through the steps in [Prerequisites](#-prerequisites).

| Mode | Tactile Source | Online/Offline | Use Case |
|---|---|---|---|
| Live | Physical sensor | Online | Interactive, short scans (<10 min) |
| Live Replay | Recorded video | Online | short scans (<10 min) |
| Offline | Recorded video | Offline | Best quality, long scans (>10 min) |

### 🟢 Mode 1: Live Reconstruction

<p align="center">
  <img src="assets/demo.gif" alt="Live GelSLAM Demo" style="width:70%;">
  <br>
  Live GelSLAM reconstruction demo, filmed live without post-processing.
</p>

Run real-time reconstruction directly on your own GelSight Mini sensor.
```bash
ros2 launch gelslam live_demo_launch.py data_dir:="/absolute/path/to/path_to_save_data" config_path:="/absolute/path/to/path_to_config"
```

  | Argument | Required | Description |
  |---|:---:|---|
  | `data_dir` | **Yes** | Where the recorded video and results will be saved |
  | `config_path` | **Yes** | Path to the GelSLAM configuration file |
  | `rendering` | No | Rendering the reconstruction progress, see [Notes](#%EF%B8%8F-notes) (default: `true`) |
  | `streamer` | No | `ffmpeg` or `cv2` (default: `ffmpeg`) |
  | `skip_background_check` | No | Skip background image confirmation, see [Notes](#%EF%B8%8F-notes) (default: `false`) |
  | `save_gelslam_states` | No | Save the GelSLAM states (default: `false`) |

- **Instructions**
  - **After launch, wait about 3 seconds before scanning to allow background capture.** Confirm the background capture is correct.
  - GelSLAM uses an FFmpeg streamer at 25 Hz by default. **On some systems, this may cause frame delay after only 40 seconds of streaming. If so, set `streamer:=cv2`, which streams at ~10 Hz. Scans slower in this case.**
  - Press `Ctrl+C` to stop scanning at any time.

- **Output**  
  After stopping, GelSLAM fuses and saves the reconstructed mesh to `<data_dir>/gelslam_online/reconstructed_mesh.ply`. For long scans, this may take up to one minute.

### Backend navigation optimization

The standard launches run `tracker_node` and `pose_graph_node` in separate processes. Keyframes and versioned 15D corrections travel over ROS topics. Tracking applies corrections at an image boundary. Navigation feedback uses corrected keyframe pose, velocity and biases to predict the next target. See [ROS process ownership and feedback](docs/slam_thread_architecture.md).

The backend stores a 15D state per keyframe: IMU pose, world velocity,
accelerometer bias and gyroscope bias. These states are the tracking keyframe
seeds. The backend has no local VI window, no direct tactile keyframe factor
and no marginalization.

Global loop correction uses a separate 15D navigation graph. Each keyframe has
full 3-axis rotation, position, world velocity and both sensor biases. Raw IMU
preintegration and bias random walk connect temporal neighbors. Contact-valid
targets also retain relative pose odometry from the frontend keyframe seeds.
Incoming IMU-only recovery pairs have no pose-odometry factor. Registered loop
poses constrain all six relative-pose components, including roll and pitch.
Each remaining component root has only its pose fixed; velocity and bias priors
remain soft observations. Global optimization does not evaluate tactile maps or
modify the stored keyframe seeds. Accepted loop measurements trigger PGO;
batches without a new loop publish seeds with the last pose, velocity and bias
corrections. Global pose, velocity and bias corrections affect output and
frontend feedback; new frontend seeds are converted back to the stored
keyframe coordinates.
The relative-pose sigmas are 0.01 rad and 0.0001 m, matching origin's 0.1 mm.
Synchronized navigation seeds, gravity, calibrated mounting and IMU intervals
are required. When global feedback changes velocity, biases or gravity attitude,
tracking requires retained IMU from the corrected keyframe to the target image.
It uses the corrected full navigation state as the fixed prediction/optimization
anchor and replaces the frontend prior only after accepting the target. A pure
yaw/XYZ frame change that preserves sensor biases and only rotates velocity
retains the existing state/prior transport path without requiring older
keyframe IMU.

Set `gelslam_config.pose_graph_config.loop_search_exclude_recent_keyframes` to
the number of preceding keyframes excluded from loop candidates. The default
`5` excludes `K_(t-1)` through `K_(t-5)` for target `K_t`; `K_(t-6)` is the
newest eligible reference. The exclusion uses DB insertion order, including
across trial boundaries, and runs before SIFT matching or NormalFlow
registration. It applies separately to every target in a queued batch.
Values must be nonnegative integers; `0` disables this candidate exclusion.
Set this in the YAML passed as `config_path`, then restart the node.

Set `gelslam_config.pose_graph_config.loop_search_min_interval_s` to the minimum
seconds between loop-search target timestamps. The default `0.0` searches every
new keyframe, including every eligible keyframe in a queued batch. For example,
`0.5` searches the first eligible target, then targets at least 0.5 data seconds
after the last searched target, whether or not that search found a loop.
Targets within the interval are skipped once; their features stay in the DB for
use as references. This does not change keyframe creation.
The search uses the navigation-state timestamp, or the image header timestamp
when navigation is unavailable, so bag replay speed does not change selection.
A new map epoch/session resets the interval. Missing or backwards timestamps
restart the interval with an immediate search. Values must be finite and
nonnegative. Set this in the YAML passed as `config_path`, then restart the node.

Add `backend_feedback:=false` to any standard launch to disable backend feedback
publication and frontend correction reception while keeping loop PGO,
map output and final backend evaluation active. The default is enabled (the bag
launch follows `backend` by default). Example:

```bash
ros2 launch gelslam bag_replay_launch.py backend:=true backend_feedback:=false
```

Rebuild `gelslam_msgs` and `gelslam` together after this interface change.
The backend requires synchronized keyframe navigation seeds and IMU intervals;
video-only input cannot supply them. Use the ROS 2 bag procedure above.
Historical pose-only snapshots require replay with IMU data before backend optimization.
Legacy v2 snapshots with direct tactile loops also require replay, because they
do not retain the registered loop measurements. New v3 snapshots retain both graphs.

### Final trajectory error colour maps

When `trajectory_evaluation.enabled` is true and at least two timestamp matches exist, the final backend trajectory is aligned using the **first matched pose** at `t0`: `A = T_GT_sensor(t0) inverse(T_map_sensor(t0))`. This single SE(3) transform is fixed for the whole selected trajectory. Both position and orientation of the anchor match GT; no later poses, position-based least-squares fit or scale fit are used. Stationary/collinear motion is supported. The anchor timestamp, frame ID and transform are saved in `trajectory_metrics.json`. The sensor is the GelSight surface frame, and the poses are resolved through final backend reference keyframes, independently of frontend State evaluation.

`gelslam_config.track_config.keep_imu_only_frames: true`로 설정하면 접촉 추적 실패로
제외하던 각 GelSight 이미지에도 IMU 예측 Frame 노드를 생성합니다. 기본값은 `false`입니다.
마지막 확정 상태 또는 적용 가능한 백엔드 보정 상태에서 해당 이미지의 IMU 시각까지
예측합니다. `tracking_status`는 `imu_prediction`입니다. 현재 이미지와 이미지 시각을
사용하며, 최종 궤적 평가는 기준 키프레임의 최종 자세로 이 Frame의 자세를 계산합니다.
확정 상태, marginal prior, 촉각 이력과 복구 키프레임 생성 주기는 유지합니다.
비접촉 전파의 유효한 Frame은 기존에도 평가에 포함됩니다.
초기화 전, IMU 구간 누락, 비접촉 전파 제한 시간 초과 또는 입력 큐에서 교체된 이미지는
추가하지 않습니다. 이 설정은 처리된 이미지에 적용됩니다. 평가 파일을 저장하려면
`trajectory_evaluation.enabled: true`도 필요합니다. 설정을 바꾼 뒤 노드를 다시 실행합니다.

The directory also contains `trajectory_translation_error.png` and `trajectory_rotation_error.png`. Each figure shows the first-pose-aligned final backend trajectory in 3D and XY, with matched GT as a dashed grey line. Pose colours show translation error in mm or SO(3) rotation-angle error in degrees. The colour bar runs from zero to the maximum error in that figure; its scale can differ between runs. Point colours use per-frame errors, and segment colours use the mean of the two endpoint errors. Gaps longer than three median sample intervals are left open. The existing `trajectory_se3_aligned.png` filename is retained for compatibility.

With `trajectory_evaluation.state_velocity.enabled: true`, the tracker records **actual frontend `State.velocity`** for published accepted contact states and no-contact predictions. The state timestamp, image timestamp, trial, raw IMU pose and unmodified velocity are saved in `state_velocity_states.jsonl`. Stale states and tactile-hold publications are excluded. This evaluation covers all recorded trials independently of backend component selection.

`keep_imu_only_frames`를 켜면 추가된 Frame의 예측 속도도 `imu_frame_prediction` 단계로
기록합니다. 이 예측 상태는 프론트엔드의 확정 상태와 구분합니다.

`state_velocity_states.jsonl`은 같은 상태의 `accel_bias`(m/s²), `gyro_bias`(rad/s),
`gravity_world_mps2`도 기록합니다. 해당 상태 시각의 IMU 측정이 버퍼에 있으면
`specific_force_imu_mps2`, `angular_velocity_imu_rps`와 `linear_acceleration_world_mps2`를
추가합니다. bias와 IMU 측정은 IMU 축을 사용합니다. 중력과 선형 가속도는 해당 trial의
상태 world 축을 사용합니다. 선형 가속도는 `R_WI (a_measured - b_a) + g_W`입니다.
이 값은 상태 시각의 진단값입니다. 구간 평균과 설정된 회전 적분법을 사용하는 실제 IMU
사전 적분의 평균 가속도와 구분해야 합니다. GT는 이 기록 계산에 사용하지 않습니다.

The simulation profile compares IMU-origin velocities using the same native GT formula as `first10_contact_state_errors_20261018` and `last_run_state_errors_20261016`: `v_WI = velocity_WS_m_s + cross(R_WI @ ideal_gyro_rad_s, p_WI - p_WS)`. The paired `trajectory_truth.npz` poses must match the received `/ground_truth/imu_pose` messages. An unrelated truth file is rejected. GT velocity is interpolated at **State.timestamp**; no extrapolation or velocity fitting is used. One fixed IMU pose transform at the first matched state of each trial maps state-world axes to GT-world axes. The configuration controls the GT pose topic, native truth file and truth time offset. This GT evaluation is independent of the IMU input replacement option.

`trajectory_velocity_error.png` colours the frontend IMU-origin trajectory by `norm(R_GT_state @ State.velocity - v_GT)` in mm/s. `trajectory_velocity_errors.csv` retains raw State velocity, transformed velocity, GT and error vectors in m/s. `trajectory_velocity_metrics.json` records RMSE, GT validation, stage/scope and trial transforms. The backend's pose export does not overwrite these files. The velocity map is not reconstructed from final position differences.

The figures use the same selected frames, matched GT and alignment as the existing final trajectory evaluation. To add these figures to existing saved results without replaying a bag, source this workspace and run:

```bash
python -m gelslam.core.trajectory_error_plot /absolute/path/to/trajectory_evaluation/sequence --label GelSLAM
```

The directory must contain `trajectory_final_aligned.csv` and `trajectory_ground_truth_matched.csv`. Existing pose CSVs and pose metrics are retained.

`trajectory_velocity_error_components.png` adds signed X/Y/Z velocity error maps in one figure. Columns correspond to components in GT world axes; the upper row shows 3D trajectories and the lower row shows XY trajectories. All panels share one symmetric colour range. Blue means negative estimate-minus-GT error, white means zero, and red means positive error. Each component title reports its RMSE.

The command above regenerates position/orientation maps. To regenerate both State velocity figures from saved states and received GT poses, use:

```bash
python -m gelslam.core.state_velocity_evaluation /absolute/path/to/evaluation/sequence \
  --truth-path /absolute/path/to/trajectory_truth.npz --time-offset-s 1.0
```

Old pose-only CSVs cannot recover State velocity. Replay is required if `state_velocity_states.jsonl` is absent.

### 🟢 Mode 2: Live Replay Reconstruction
Run real-time reconstruction by replaying a recorded GelSight video. Tactile images are streamed at the sensor’s native frame rate to simulate live input.
```bash
ros2 launch gelslam live_replay_demo_launch.py data_dir:="/absolute/path/to/path_to_recorded_data" config_path:="/absolute/path/to/path_to_config"
```

  | Argument | Required | Description |
  |---|:---:|---|
  | `data_dir` | **Yes** | Containing the GelSight video and where results will be saved |
  | `config_path` | **Yes** | Path to the GelSLAM configuration file |
  | `rendering` | No | Rendering the reconstruction progress, see [Notes](#%EF%B8%8F-notes) (default: `true`) |
  | `skip_background_check` | No | Skip background image confirmation, see [Notes](#%EF%B8%8F-notes) (default: `true`) |
  | `save_gelslam_states` | No | Save the GelSLAM states (default: `false`) |

- **Instructions**
  - Place the GelSight video in `data_dir` and name it `gelsight.avi`. **Ensure no contact occurs during the first 10 frames, which are used as GelSight background.**
  - Once the video stream ended, the process will terminate itself. Otherwise, you may press **Ctrl+C** to stop at any time.

- **Output**  
  After completion (or early termination), GelSLAM fuses and saves the reconstructed mesh to: `<data_dir>/gelslam_online/reconstructed_mesh.ply`. For long videos, this may take up to a minute.

### 🟢 Mode 3: Offline Reconstruction
Run offline reconstruction on a recorded GelSight video.

```bash
ros2 run gelslam gelslam_offline -d "/absolute/path/to/path_to_data" -c "/absolute/path/to/path_to_config"
```

  | Argument | Required | Description |
  |---|:---:|---|
  | `data_dir` | **Yes** | Containing the GelSight video and where results will be saved |
  | `config_path` | **Yes** | Path to the GelSLAM configuration file |
  | `rendering` | No | Rendering the reconstruction progress, see [Notes](#%EF%B8%8F-notes) (default: `false`) |
  | `save_gelslam_states` | No | Save the GelSLAM states (default: `false`) |

- **Instructions**
  - Place the GelSight video in `data_dir` and name it `gelsight.avi`. **Ensure no contact occurs during the first 10 frames, which are used as GelSight background.**
  - The process terminates automatically once the reconstruction ends.
  - For extremely long videos (>30 minutes), memory usage may become a concern, as the recorded video itself can be several gigabytes.

- **Output**  
  After completion, the reconstructed mesh is saved to: `<data_dir>/gelslam_offline/reconstructed_mesh.ply`.

### ⚠️ Notes

- GelSLAM requires a background image captured from the first few frames. New users are asked to confirm the background image before continuing. Set `skip_background_check:=true` to skip background confirmation.
- The `rendering` option is for displaying the reconstruction progress and is not part of the GelSLAM algorithm. When the meshes grow large (which can reach gigabytes), rendering it can slow down the system and affects GelSLAM performance.
- If `rendering:=true`, the rendering resolution is automatically reduced as the mesh grows, but the underlying reconstructed mesh is kept at full resolution.
- For very large objects that requires scans longer than 10 minutes, we recommend recording multiple Live scans, stitching the videos, and running offline reconstruction for best quality.

---

### 🔁 Reproduce Paper Results
To reproduce the reconstruction results for the 15 objects in our [paper](https://arxiv.org/abs/2508.15990), follow these steps:

1. **Download the dataset**:  
Download the [dataset](https://huggingface.co/datasets/joehjhuang/GelSLAM_dataset) and unzip `dataset.zip`.

2. **Run Offline Reconstruction**:  
For each object in the `reconstruction_dataset`, run
```bash
ros2 run gelslam gelslam_offline -d "/absolute/path/to/dataset_dir"/reconstruction_dataset/"/absolute/path/to/object_name" -c "/absolute/path/to/dataset_dir"/reconstruction_dataset/"/absolute/path/to/object_name"/config.yaml
```

This original video-only reproduction procedure needs a compatible historical backend; it does not run the current 15D backend from video alone. All objects share the same `gelslam_config` in the configuration file, but different calibrations, as five GelPads were used across experiments.

---

### 🔧 Tools

<p align="center">
  <img src="assets/pose_graph.jpg" alt="Pose Graph" style="width:70%; height:auto;">
  <br>
  Pose graph. Black nodes: keyframes; green nodes: coverage keyframes; red edges: pairwise pose constraints.
</p>

We provide a tool to visualize the final pose graph of GelSLAM. When [running GelSLAM](#-running-gelslam), enable state saving by setting `save_gelslam_states:=true`, then run the following to visualize the pose graph:
```bash
ros2 run gelslam visualize_pose_graph -d "/absolute/path/to/path_to_data" -c "/absolute/path/to/path_to_config" -m "/absolute/path/to/method_name"
```

  | Argument | Required | Description |
  |---|:---:|---|
  | `data_dir` | **Yes** | Containing the GelSight video and where results will be saved |
  | `config_path` | **Yes** | Path to the GelSLAM configuration file |
  | `method_name` | **Yes** |  `gelslam_online` (Mode 1 and 2) or `gelslam_offline` (Mode 3) |

---

### 🚧 Update Roadmap

We plan to add the following features to the repository:
- An online pose estimation mode for GelSLAM.

---

### 📖 Cite GelSLAM
If you find this package useful, please consider citing our arXiv paper:
```
@misc{huang2025gelslam,
      author={Hung-Jui Huang and Mohammad Amin Mirzaee and Michael Kaess and Wenzhen Yuan},
      title={GelSLAM: A Real-time, High-Fidelity, and Robust 3D Tactile SLAM System}, 
      year={2025},
      eprint={2508.15990},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.15990}, 
}
```
