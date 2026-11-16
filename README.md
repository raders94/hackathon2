# PORTCAST | 2026 AI 해커톤 · 7개 터미널 기상영향/운영학습

**Streamlit Community Cloud 업로드 준비 완료한 저장소 루트 파일 모음**입니다. 2020–2025 선박·기상 분석, CatBoost 예상 작업률, 개별 예측 SHAP, 2025 미학습 악천후 성능, 작업량 보정 모델 B 및 2022–2024 운영학습 추세, 디지털 트윈 후속 검증 제안을 통합했습니다.

## 기능별 탭

1. **AI 즉시 예측:** BCT/BNCT/HPNT/PNC/PNIT/BPTC_G/BPT_S, 미래 가상날짜, 풍속·파고·시정·혼잡, CatBoost 수치와 SHAP, 결과 CSV
2. **7터미널 기상영향:** 7단계 회귀·GT 통제·첫6시간 강풍 노출 비교, 집계자료 그래프
3. **모델 B · 운영학습:** 운영변수 보정 후 기상관련 작업효율 차이, BCT 다중검증, 기상변수 추가 전/후 예측오차, 모델 B SHAP, 2022–24 연도별 개선속도 비교
4. **디지털 트윈 제안:** 희소 악천후 반복훈련 4단계 개념, 사용자 가정 회복 목표 계산. **물리·운영 시뮬레이터 미구현**
5. **현장 실증 · 모델 신뢰도:** 90일 계획, 터미널별 2025 예측 R², 모델 평가의 한계
6. **SHAP · 악천후별 검증:** 2025 미학습 데이터, P90/P95 및 기상요소별 MAE/WAPE/R², 표본수·관측 누락 점검

## GitHub → Streamlit 배포

**중요:** ZIP 파일을 저장소에 1개 올리는 방식이 아닙니다. ZIP을 PC에 풀고 **`app.py`, `assets/`, `src/`, `docs/`, `.streamlit/`, `requirements.txt` 등 내용물 자체**를 저장소 최상위 경로에 올려야 합니다.

### GitHub 웹을 사용한 업로드

1. [GitHub 새 저장소](https://github.com/new)에서 **빈 저장소** 생성. 공개 배포 전 데이터·학습모형 재배포 권한 확인. 확인 전에는 공개 저장소 사용을 보류하세요.
2. 이 ZIP을 압축 해제. `app.py`가 바로 보이는 폴더의 **모든 파일·폴더를 선택**하고 GitHub의 `Add file → Upload files`로 업로드합니다. 폴더 구조가 유지되어야 합니다.
3. 저장소 최상위에 `app.py`와 `requirements.txt`, `.streamlit/config.toml`, `assets/*.cbm`, `src/logic.py`가 존재하는지 확인.
4. [Streamlit Community Cloud](https://share.streamlit.io)에서 GitHub 계정을 연결 → **Create app** → 기존 저장소 선택 → 브랜치 `main` → **Main file path `app.py`** → Advanced Settings에서 Python **3.12** 선택 → Deploy.
5. 배포 후 첫 탭에서 BCT 가상 시나리오 예측 및 SHAP 그래프, 세 번째 탭에서 모델 B와 연도별 추세, 여섯 번째 탭에서 검증표를 확인하고 공개 주소를 공유합니다.

### Git CLI (Git 설치한 PC에서)

```bash
git init
git add .
git commit -m "PORTCAST hackathon integrated dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPO.git
git push -u origin main
```

원격 저장소 주소는 자신의 GitHub 주소로 변경합니다. 저장소는 빈 상태여야 기존 README와 충돌하지 않습니다.

### Windows 로컬 실행

Windows에서 `RUN_LOCAL_WINDOWS.bat` 실행. Python 3.12 설치가 필요합니다. 또는:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## 공개 배포 및 정확성 주의

- **원자료 제공기관의 공개·클라우드 업로드·학습모형 재배포 권한을 직접 확인하세요.** 이 ZIP은 선박별 원시 데이터(`vessel_history.csv.gz`)와 시간별 관측 이력(`weather_history.csv.gz`)을 제외하고 **통계집계 CSV와 학습모형**만 포함합니다. 그러나 가공 집계나 학습모형에도 사용권 문제가 있을 수 있습니다.
- `.streamlit/secrets.toml`, 원자료 파일, 계정/선박 식별정보, 개인식별정보를 GitHub에 포함하지 마세요.
- 미래 날짜는 실제 기상예보 API와 연결되지 않은 가상 시나리오입니다. 예측 수치는 작업량/전체 접안후 체류시간에 기반하며 실시간 크레인 Moves/h가 아닙니다.
- 모델 A의 개별 SHAP과 모델 B의 잔차분석 SHAP은 다른 모델이며, 회귀의 기상 관련 효율차이는 인과추정치가 아닙니다.
- 2025 신항의 기상 완전관측률이 낮고 이벤트시각 관측 선택성이 있어 검증 결과의 일반화에는 주의가 필요합니다.
- 디지털 트윈 훈련과 운영정책 최적화, 생산성 개선, 실제 운영시스템 제어 기능은 **현재 미구현**입니다.

## 폴더 안내

```text
PORTCAST/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
├── src/logic.py / story.py / explainability.py / digital_twin_lab.py
├── assets/        (집계 CSV + CatBoost 모델 2개)
├── docs/          (SHAP·검증 보고서·프로젝트 소개글·AI 활용내역)
├── RUN_LOCAL_WINDOWS.bat
├── check_release.py
└── README.md
```

## 제출용 텍스트

- [해커톤 소개글·AI 활용내역 수정본](docs/해커톤_소개글_AI활용내역_수정본.md) (소개글 500자 이내)
- [분석 결과 해석 주의 요약](docs/개선제안_분석주의_요약.md)
- 기존 `docs/VALIDATION_REPORT_KO.md` 및 `docs/SHAP_INTERPRETATION_KO.md`

## 테스트 범위

`check_release.py`는 필수 파일·모델 로드·BCT 가상 예측·SHAP 합산 검산·분석 요약치 일치 여부를 점검합니다. 본 패키지는 소스정적검사 및 계산용 모의실행을 수행했으며, **실제 Cloud 배포는 계정 연결·권한 확인 후 사용자가 진행해야 합니다**. `streamlit run app.py`의 실제 브라우저 구동은 별도 확인하세요.
