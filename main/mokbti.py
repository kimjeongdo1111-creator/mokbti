# -*- coding: utf-8 -*-
"""
=============================================================
Voice-BTI Studio: 올인원 단일 통합 파이썬 애플리케이션
=============================================================
1. 직접 구현한 Radix-2 Cooley-Tukey FFT / IFFT / STFT / ISTFT
2. 스펙트럼 차감법(Spectral Subtraction) 노이즈 캔슬링
3. pYIN 알고리즘 기반 기본주파수(F0) 추적 및 5대 목BTI 분석
4. Streamlit 기반 고대비 라이트 테마 웹 대시보드 및 수학적 모델링
=============================================================
실행 방법: streamlit run app.py
=============================================================
"""

import io
import os
import base64
import subprocess
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import matplotlib.patheffects as patheffects
from scipy.ndimage import minimum_filter1d
import scipy.fft as sfft
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None
import streamlit as st


def get_image_base64(filename):
    """image 폴더 내의 png 파일을 base64 데이터 URI로 인코딩하여 반환"""
    if not filename:
        return ""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "image", filename),
        os.path.join(os.getcwd(), "image", filename),
        os.path.join("image", filename)
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return ""


# =========================================================
# [1단계] 한글 폰트(맑은 고딕) 자동 등록 (폰트 깨짐 100% 방지)
# =========================================================
font_path = "C:/Windows/Fonts/malgun.ttf"
try:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        KOREAN_FONT = font_prop.get_name()
    else:
        # Streamlit Cloud(리눅스) 등 C: 드라이브가 없는 환경
        KOREAN_FONT = 'NanumGothic'
    plt.rcParams['font.family'] = KOREAN_FONT
except Exception:
    KOREAN_FONT = 'NanumGothic'
    plt.rcParams['font.family'] = KOREAN_FONT

plt.rcParams['font.sans-serif'] = [KOREAN_FONT, 'NanumGothic', 'Malgun Gothic', 'Apple SD Gothic Neo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# FFmpeg 경로 시스템 연동
if imageio_ffmpeg is not None:
    try:
        _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
        if _ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass



# =========================================================
# [2단계] 고효율 라이브러리(scipy.fft) 기반 FFT & STFT 구현
# =========================================================
def fft_fast(x, n=None, axis=0):
    """scipy.fft 기반의 최고 효율 초고속 고속 푸리에 변환 (FFT)"""
    return sfft.fft(x, n=n, axis=axis)


def ifft_fast(X, n=None, axis=0):
    """scipy.fft 기반의 최고 효율 역 고속 푸리에 변환 (IFFT)"""
    return sfft.ifft(X, n=n, axis=axis)


# 이전 버전 및 외부 참조 호환용 별칭
fft_cooley_tukey = fft_fast
ifft_cooley_tukey = ifft_fast


def custom_stft(y, n_fft=2048, hop_length=512, window='hann'):
    """고효율 라이브러리(scipy.fft) 기반 단시간 푸리에 변환 (STFT)"""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    win = np.hanning(n_fft)
    pad_amount = n_fft // 2
    y_padded = np.pad(y, pad_amount, mode='reflect')

    num_frames = 1 + (len(y_padded) - n_fft) // hop_length
    if num_frames <= 0:
        y_padded = np.pad(y_padded, (0, n_fft), mode='constant')
        num_frames = 1

    shape = (n_fft, num_frames)
    strides = (y_padded.strides[0], y_padded.strides[0] * hop_length)
    frames = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)

    windowed_frames = frames * win[:, None]
    # C로 최적화된 scipy.fft.rfft로 단시간 주파수 스펙트럼 고속 계산
    spectrum = sfft.rfft(windowed_frames, n=n_fft, axis=0)
    return spectrum


def custom_istft(S, hop_length=512, length=None, window='hann'):
    """고효율 라이브러리(scipy.fft) IFFT 및 Overlap-Add(OLA)를 이용한 시간 영역 음성 신호 복원"""
    n_freq, num_frames = S.shape
    n_fft = (n_freq - 1) * 2

    # scipy.fft.irfft를 통해 대칭 켤레 복소수 자동 처리 및 실수 신호 고속 복원
    time_frames = sfft.irfft(S, n=n_fft, axis=0)
    win = np.hanning(n_fft)[:, None]
    time_frames = time_frames * win

    expected_signal_len = n_fft + (num_frames - 1) * hop_length
    y_out = np.zeros(expected_signal_len, dtype=np.float64)
    win_sq_acc = np.zeros(expected_signal_len, dtype=np.float64)
    win_sq = (win ** 2).flatten()

    for i in range(num_frames):
        start = i * hop_length
        end = start + n_fft
        y_out[start:end] += time_frames[:, i]
        win_sq_acc[start:end] += win_sq

    nonzero = win_sq_acc > 1e-8
    y_out[nonzero] /= win_sq_acc[nonzero]

    pad_amount = n_fft // 2
    y_trimmed = y_out[pad_amount:]
    if length is not None:
        if len(y_trimmed) < length:
            y_trimmed = np.pad(y_trimmed, (0, length - len(y_trimmed)), mode='constant')
        else:
            y_trimmed = y_trimmed[:length]
    return y_trimmed


# =========================================================
# =========================================================
# [3단계] 오디오 디코딩, 노이즈 캔슬링 & 목BTI 분류기
# =========================================================
VOICE_BTI_PROFILES = {
    "하이톤형": {
        "id": "high_tone", "name": "하이톤형", "en_name": "High-Tone Sparkling", "color": "#00E5FF",
        "image_file": "dolphin.png",
        "tagline": "청량함 100%! 분위기를 단숨에 반전시키는 청아한 고음",
        "summary": "밝고 통통 튀며 귀에 쏙쏙 박히는 에너제틱한 고음 보이스입니다. 대화방의 분위기 메이커 역할을 톡톡히 하며 청중의 집중을 즉각 이끌어냅니다.",
        "traits": ["맑고 시원하게 뻗어나가는 청량한 울림", "말끝에 기분 좋은 긍정의 에너지가 실려 있음", "사람들의 이목을 사로잡는 탁월한 시선 집중력"],
        "charm_points": "듣기만 해도 기분이 상쾌해지는 비타민 같은 청아함과 사랑스러움",
        "best_careers": ["콘텐츠 크리에이터", "라이브 커머스 쇼호스트", "애니메이션/게임 성우", "스피치 코치", "행사 전문 MC"],
        "duo_match": "동굴저음형 (반대 매력의 완벽한 하모니를 이루는 극과 극 케미)",
        "signature_phrase": "“자, 다들 주목~! 오늘 정말 신나고 재밌는 일이 생겼어요!”"
    },
    "비타민형": {
        "id": "vitamin", "name": "비타민형", "en_name": "Vitamin Bright", "color": "#10B981",
        "image_file": "lemon.png",
        "tagline": "호감도 급상승! 친근하고 산뜻한 생기발랄 보이스",
        "summary": "자연스러운 밝음과 생기가 묻어나는 상큼한 중고음 보이스입니다. 누구에게나 다정하고 친근하게 다가가는 독보적인 친화력을 자랑합니다.",
        "traits": ["적절한 억양과 산뜻한 리듬감으로 지루할 틈이 없는 스피치", "듣는 이에게 기분 좋은 온기와 긍정적 영감을 불어넣음", "어떤 대화에서도 어색함을 없애는 마법 같은 친화력"],
        "charm_points": "첫마디만 들어도 무장해제되는 친절함과 밝은 미소가 연상되는 음색",
        "best_careers": ["피플팀(HR) 리크루터", "브랜드 마케터", "고객 경험(CX) 전략가", "방송 리포터", "프로덕트 매니저(PM)"],
        "duo_match": "ASMR형 (활기와 차분함이 만나 균형 잡힌 최고의 편안함 선사)",
        "signature_phrase": "“안녕하세요 여러분! 오늘 하루도 기분 좋게 출발해 볼까요?”"
    },
    "내레이션형": {
        "id": "narration", "name": "내레이션형", "en_name": "Narration Classic", "color": "#8B5CF6",
        "image_file": "nick.png",
        "tagline": "신뢰도 200%! 귀에 꽂히는 또렷하고 안정감 있는 정석 톤",
        "summary": "차분하고 지적이며 전달력이 매우 우수한 표준적이고 균형 잡힌 보이스입니다. 사람들에게 확신과 안정감을 심어주며 프레젠테이션과 설득에 강합니다.",
        "traits": ["불필요한 떨림 없이 곧고 명확하게 꽂히는 발음과 톤", "정보 전달력이 뛰어나 듣는 이가 오래 들어도 피로하지 않음", "지적이고 프로페셔널한 인상을 주는 안정적인 피치"],
        "charm_points": "단번에 사람을 사로잡는 신뢰감과 믿음직스러운 어른의 분위기",
        "best_careers": ["뉴스 앵커 / 아나운서", "다큐멘터리 내레이터", "스타트업 대표 / IR 전문가", "전문 컨설턴트", "변호사 / 법조인"],
        "duo_match": "비타민형 (신뢰감에 발랄함을 더해주는 환상의 비즈니스 파트너)",
        "signature_phrase": "“지금부터 전해드릴 소식에 주목해 주시기 바랍니다.”"
    },
    "ASMR형": {
        "id": "asmr", "name": "ASMR형", "en_name": "ASMR Cozy", "color": "#64748B",
        "image_file": "microphone.png",
        "tagline": "귓가를 감싸는 포근함! 마음을 녹여주는 나직하고 감미로운 힐링 톤",
        "summary": "부드럽고 나직하게 속삭이듯 다가와 지친 하루의 피로를 풀어주는 힐링 보이스입니다. 따뜻한 공명과 숨결이 어우러져 듣는 이에게 심리적 안정감을 선물합니다.",
        "traits": ["자극 없이 귓가에 스며드는 따뜻하고 둥근 톤", "숨소리와 잔잔한 울림이 편안하게 조화를 이룸", "새벽 감성을 자극하는 포근하고 몽환적인 감미로움"],
        "charm_points": "잠들기 전 침대 맡에서 듣고 싶은 고막 힐링과 다정한 속삭임",
        "best_careers": ["오디오북 전문 낭독가", "심리 상담사 / 멘탈 코치", "사운드 디자이너", "심야 라디오 진행자", "웰니스 디렉터"],
        "duo_match": "내레이션형 (지적인 설명과 감성적 위로가 어우러지는 완벽한 콤비)",
        "signature_phrase": "“오늘 하루도 정말 고생 많았어요. 편안한 밤 보내세요.”"
    },
    "동굴저음형": {
        "id": "cave_bass", "name": "동굴저음형", "en_name": "Cave Deep Bass", "color": "#F59E0B",
        "image_file": "linggom.png",
        "tagline": "심장을 울리는 치명적 중후함! 압도적인 카리스마의 딥 베이스",
        "summary": "가슴 깊은 곳에서 울려 퍼지는 묵직하고 깊은 동굴 저음 보이스입니다. 흉성의 풍부한 울림으로 범접할 수 없는 카리스마와 짙은 중후미를 풍깁니다.",
        "traits": ["심장을 은은하게 울리는 풍부한 저주파수 공명", "묵직한 무게감과 함께 묻어나는 압도적인 분위기", "짧은 한마디로도 방 안의 공기를 장악하는 묵직한 카리스마"],
        "charm_points": "귓가를 전율케 하는 묵직한 베이스 울림과 깊이를 알 수 없는 중후한 매력",
        "best_careers": ["영화 예고편 / CF 성우", "오디오 브랜드 디렉터", "클래식 성악가(베이스/바리톤)", "다큐멘터리 해설가", "오디오북 보이스 액터"],
        "duo_match": "하이톤형 (고음의 화려함과 저음의 웅장함이 만나는 드라마틱 앙상블)",
        "signature_phrase": "“그 누구도 예상하지 못한 진실이, 지금 시작됩니다.”"
    }
}
ORDERED_TYPES = ["하이톤형", "비타민형", "내레이션형", "ASMR형", "동굴저음형"]


def decode_audio_bytes(raw_bytes):
    """모든 오디오 포맷(M4A, MP3, WAV, WebM 등)을 float32 신호로 자동 디코딩"""
    if not raw_bytes or len(raw_bytes) == 0:
        raise ValueError("오디오 데이터가 비어 있습니다.")
    try:
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
        return y.astype(np.float32), sr
    except Exception:
        pass
    if imageio_ffmpeg is not None:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            process = subprocess.Popen(
                [ffmpeg_exe, "-y", "-i", "pipe:0", "-vn", "-ac", "1", "-f", "wav", "pipe:1"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            wav_out, _ = process.communicate(input=raw_bytes)
            y, sr = sf.read(io.BytesIO(wav_out), dtype='float32')
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            return y.astype(np.float32), sr
        except Exception:
            pass
    raise ValueError("오디오 형식을 디코딩할 수 없습니다. 정상적인 MP3 또는 WAV 파일인지 확인해 주세요.")


def spectral_subtraction_detailed(y, sr, noise_duration=0.5, n_fft=2048, hop_length=512, alpha=1.5, beta=0.02):
    """스펙트럼 차감법 및 알고리즘 시각화 중간값 반환"""
    S = custom_stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)
    phase = np.angle(S)

    noise_frames = max(1, min(int(noise_duration * sr / hop_length), magnitude.shape[1]))
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1)

    subtracted_magnitude = magnitude - (alpha * noise_profile[:, None])
    clean_magnitude = np.maximum(subtracted_magnitude, beta * noise_profile[:, None])

    clean_S = clean_magnitude * np.exp(1j * phase)
    y_denoised = custom_istft(clean_S, hop_length=hop_length, length=len(y))

    mid_frame = min(magnitude.shape[1] // 2, magnitude.shape[1] - 1)
    intermediates = {
        'noise_profile': noise_profile,
        'magnitude_original': magnitude,
        'clean_magnitude': clean_magnitude,
        'n_fft': n_fft, 'hop_length': hop_length,
        'noise_frames': noise_frames, 'mid_frame': mid_frame,
        'sr': sr, 'alpha': alpha, 'beta': beta,
    }
    return y_denoised, intermediates


def spectral_subtraction_minimum_statistics_detailed(y, sr, window_duration=1.5, n_fft=2048, hop_length=512, alpha=1.5, beta=0.02, bias_factor=1.5):
    """동적 최소 통계법 노이즈 차감"""
    S = custom_stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)
    phase = np.angle(S)

    window_frames = max(1, min(int(np.ceil((window_duration * sr) / hop_length)), magnitude.shape[1]))
    min_noise = minimum_filter1d(magnitude, size=window_frames, axis=1, mode='nearest')
    noise_profile_dynamic = min_noise * bias_factor

    subtracted_magnitude = magnitude - (alpha * noise_profile_dynamic)
    clean_magnitude = np.maximum(subtracted_magnitude, beta * noise_profile_dynamic)

    clean_S = clean_magnitude * np.exp(1j * phase)
    y_denoised = custom_istft(clean_S, hop_length=hop_length, length=len(y))

    mid_frame = min(magnitude.shape[1] // 2, magnitude.shape[1] - 1)
    intermediates = {
        'noise_profile': noise_profile_dynamic[:, mid_frame],
        'magnitude_original': magnitude,
        'clean_magnitude': clean_magnitude,
        'n_fft': n_fft, 'hop_length': hop_length,
        'noise_frames': window_frames, 'mid_frame': mid_frame,
        'sr': sr, 'alpha': alpha, 'beta': beta,
    }
    return y_denoised, intermediates


def extract_pitch_and_voicing(y, sr, hop_length=512, fmin=55, fmax=550):
    """
    pYIN 알고리즘 기반 F0 기본주파수 추출 + 적응형 VAD(발화 감지) + 주파수 정체(변화량 0) 노이즈 필터링
    """
    max_amp = np.max(np.abs(y))
    y_norm = y / max_amp if max_amp > 1e-6 else y

    rms = librosa.feature.rms(y=y_norm, frame_length=2048, hop_length=hop_length)[0]

    f0, voiced_flag, _ = librosa.pyin(
        y_norm, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop_length, fill_na=np.nan
    )

    N = len(f0)
    rms_aligned = rms[:N]
    max_rms = np.max(rms_aligned) if len(rms_aligned) > 0 else 1.0

    # 1. 적응형 VAD: 실제 말이 나오는 타이밍만 추출 (침묵 및 숨소리/배경잡음 컷오프)
    vad_mask = (rms_aligned >= max(0.025 * max_rms, 0.003))

    # 2. 주파수 변화량 0 구간 필터링: 연속 3프레임 이상 주파수가 고정된 기계적 험 노이즈 및 정체 구간 배제
    is_flat = np.zeros(N, dtype=bool)
    for i in range(1, N - 1):
        if np.abs(f0[i] - f0[i-1]) < 0.08 and np.abs(f0[i+1] - f0[i]) < 0.08:
            is_flat[i-1] = True
            is_flat[i] = True
            is_flat[i+1] = True

    # 3. 유효 발화 마스크 통합 (유성음 + NaN 아님 + VAD 통과 + 주파수 변화성 유지 + 인간 발화 주파수 범위 60~500Hz)
    valid_mask = voiced_flag & (~np.isnan(f0)) & vad_mask & (~is_flat) & (f0 >= 60.0) & (f0 <= 500.0)

    # 발화 음량이 작거나 짧아서 필터링이 과도할 경우의 보정 fallback
    if np.sum(valid_mask) < 5:
        valid_mask = voiced_flag & (~np.isnan(f0)) & (rms_aligned >= 0.005 * max_rms) & (f0 >= 60.0) & (f0 <= 500.0)

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    valid_pitches = f0[valid_mask]
    return f0, valid_mask, times, valid_pitches


def classify_voice_bti(valid_pitches, th_high=255.0, th_vitamin=190.0, th_narration=135.0, th_asmr=95.0):
    """5대 목BTI 유형 점유율(%) 및 정량 지표 산출"""
    if len(valid_pitches) == 0:
        return {"success": False, "message": "목소리 주파수가 감지되지 않았습니다. 조금 더 크고 또렷하게 말씀해 주세요."}

    total = len(valid_pitches)
    counts = {
        "하이톤형": int(np.sum(valid_pitches >= th_high)),
        "비타민형": int(np.sum((valid_pitches >= th_vitamin) & (valid_pitches < th_high))),
        "내레이션형": int(np.sum((valid_pitches >= th_narration) & (valid_pitches < th_vitamin))),
        "ASMR형": int(np.sum((valid_pitches >= th_asmr) & (valid_pitches < th_narration))),
        "동굴저음형": int(np.sum(valid_pitches < th_asmr))
    }
    percentages = {t: round((counts[t] / total) * 100.0, 1) for t in ORDERED_TYPES}
    diff = 100.0 - sum(percentages.values())
    max_k = max(percentages, key=percentages.get)
    percentages[max_k] = round(percentages[max_k] + diff, 1)

    sorted_t = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    top_type = sorted_t[0][0]
    sub_type = sorted_t[1][0] if len(sorted_t) > 1 and sorted_t[1][1] > 10.0 else None

    mean_f0 = float(np.mean(valid_pitches))
    try:
        note_mean = librosa.hz_to_note(mean_f0)
    except Exception:
        note_mean = "N/A"

    return {
        "success": True, "percentages": percentages, "counts": counts, "top_type": top_type, "sub_type": sub_type,
        "mean_f0": round(mean_f0, 1), "median_f0": round(float(np.median(valid_pitches)), 1),
        "min_f0": round(float(np.percentile(valid_pitches, 5)), 1), "max_f0": round(float(np.percentile(valid_pitches, 95)), 1),
        "std_f0": round(float(np.std(valid_pitches)), 1), "note_mean": note_mean,
        "thresholds": {"th_high": th_high, "th_vitamin": th_vitamin, "th_narration": th_narration, "th_asmr": th_asmr}
    }


# =========================================================
# [4단계] Matplotlib 모바일 최적화 고대비 차트
# =========================================================
def plot_voice_bti_donut(bti_res):
    """Plotly 기반 모바일 최적화 도넛 차트 (불필요한 글자 없이 유형명 + %만 심플하게 표시)"""
    import plotly.graph_objects as go

    percentages = bti_res['percentages']
    top_type    = bti_res['top_type']
    top_pct     = percentages[top_type]
    top_color   = VOICE_BTI_PROFILES[top_type]['color']

    labels = ORDERED_TYPES
    values = [max(percentages.get(t, 0), 0.0) for t in labels]
    colors = [VOICE_BTI_PROFILES[t]['color'] for t in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.66,
        marker=dict(
            colors=colors,
            line=dict(color='#FFFFFF', width=3)
        ),
        textinfo='none',
        hovertemplate='<b>%{label}</b>: %{value:.1f}%<extra></extra>',
        sort=False,
        direction='clockwise',
        rotation=90,
    )])

    # 글자 겹침을 원천 차단하기 위해 유형명과 퍼센트를 2개의 독립된 어노테이션(y 좌표 분리)으로 배치
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=f"<b>{top_type}</b>",
                font=dict(size=24, color=top_color, family='Pretendard, Apple SD Gothic Neo, sans-serif'),
                x=0.5, y=0.57,
                xref='paper', yref='paper',
                showarrow=False,
                align='center',
            ),
            dict(
                text=f"<b>{top_pct}%</b>",
                font=dict(size=20, color='#0F172A', family='Pretendard, Apple SD Gothic Neo, sans-serif'),
                x=0.5, y=0.43,
                xref='paper', yref='paper',
                showarrow=False,
                align='center',
            )
        ]
    )
    return fig



def plot_voice_spectrum_bar(valid_pitches, bti_res):
    """Plotly 기반 모바일 최적화 주파수 스펙트럼 내 음역대 영역 분포 바 (서버 폰트 무관 한글 100% 렌더링)"""
    import plotly.graph_objects as go

    x_min, x_max = 60.0, 360.0
    th = bti_res["thresholds"]
    zones = [
        ("동굴저음형", "< 95Hz", x_min, th["th_asmr"], "#8B5CF6", 0.16),
        ("ASMR형", "95-135Hz", th["th_asmr"], th["th_narration"], "#A855F7", 0.18),
        ("내레이션형", "135-190Hz", th["th_narration"], th["th_vitamin"], "#3B82F6", 0.20),
        ("비타민형", "190-255Hz", th["th_vitamin"], th["th_high"], "#10B981", 0.20),
        ("하이톤형", "> 255Hz", th["th_high"], x_max, "#06B6D4", 0.18),
    ]

    fig = go.Figure()

    annotations = []
    shapes = []

    # 1. 5대 음역대 배경 밴드 및 라벨
    for name, sub, z_start, z_end, color, alpha in zones:
        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=z_start, x1=z_end,
            y0=0.15, y1=0.85,
            fillcolor=color,
            opacity=alpha,
            line=dict(color="#CBD5E1", width=1, dash="dash")
        ))
        mid = (z_start + z_end) / 2
        annotations.append(dict(
            x=mid, y=0.50,
            xref="x", yref="y",
            text=f"<b>{name}</b><br><span style='font-size:10px; color:#475569;'>({sub})</span>",
            showarrow=False,
            align="center",
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            borderwidth=1,
            borderpad=3,
            font=dict(size=11, color="#1E293B", family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif")
        ))

    min_f0 = max(x_min, bti_res["min_f0"])
    max_f0 = min(x_max, bti_res["max_f0"])
    if max_f0 - min_f0 < 8.0:
        max_f0 = min(x_max, min_f0 + 8.0)
    mean_f0 = bti_res["mean_f0"]
    note_clean = bti_res['note_mean'].replace('♯', '#')

    # 2. 사용자 발화 음역폭 (핑크 밴드)
    shapes.append(dict(
        type="rect",
        xref="x", yref="y",
        x0=min_f0, x1=max_f0,
        y0=0.18, y1=0.82,
        fillcolor="rgba(244, 63, 94, 0.25)",
        line=dict(color="#E11D48", width=2.5)
    ))

    # 3. 평균 기본주파수 세로선
    shapes.append(dict(
        type="line",
        xref="x", yref="y",
        x0=mean_f0, x1=mean_f0,
        y0=0.06, y1=0.94,
        line=dict(color="#D97706", width=3.5)
    ))

    # 4. 평균 기본주파수 원형 마커
    fig.add_trace(go.Scatter(
        x=[mean_f0], y=[0.50],
        mode="markers",
        marker=dict(size=14, color="#D97706", line=dict(color="#FFFFFF", width=3)),
        hoverinfo="skip",
        showlegend=False
    ))

    # 5. 상단 평균주파수 플래그
    annotations.append(dict(
        x=mean_f0, y=1.04,
        xref="x", yref="y",
        text=f"<b>평균 기본주파수: {mean_f0:.1f}Hz ({note_clean})</b>",
        showarrow=False,
        align="center",
        bgcolor="#FEF3C7",
        bordercolor="#F59E0B",
        borderwidth=1.5,
        borderpad=4,
        font=dict(size=12, color="#92400E", family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif")
    ))

    # 6. 하단 최저/최고 음역 라벨
    annotations.append(dict(
        x=min_f0, y=-0.06,
        xref="x", yref="y",
        text=f"<b>최저 {min_f0:.0f}Hz</b>",
        showarrow=False,
        align="center",
        font=dict(size=11, color="#BE123C", family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif")
    ))
    annotations.append(dict(
        x=max_f0, y=-0.06,
        xref="x", yref="y",
        text=f"<b>최고 {max_f0:.0f}Hz</b>",
        showarrow=False,
        align="center",
        font=dict(size=11, color="#BE123C", family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif")
    ))

    ticks = [60, 95, 135, 190, 255, 300, 360]
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            range=[x_min - 2, x_max + 2],
            tickmode="array",
            tickvals=ticks,
            ticktext=[f"<b>{t}Hz</b>" for t in ticks],
            tickfont=dict(size=11, color="#334155", family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif"),
            title=dict(
                text="<b>주파수 대역 (Frequency Hz)</b>",
                font=dict(size=12, color="#475569", family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif")
            ),
            showgrid=False,
            zeroline=False,
            linecolor="#E2E8F0",
            linewidth=1.2,
        ),
        yaxis=dict(
            range=[-0.18, 1.25],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        height=240,
        margin=dict(l=10, r=10, t=28, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
        showlegend=False,
    )
    return fig




def plot_pitch_contour(times, f0, valid_mask, bti_res):
    fig, ax = plt.subplots(figsize=(16, 4.0))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    t_min = times[0] if len(times) > 0 else 0
    t_max = times[-1] if len(times) > 0 else 1

    ax.axhspan(255, 400, color='#06B6D4', alpha=0.10, label="하이톤 대역 (> 255Hz)")
    ax.axhspan(190, 255, color='#10B981', alpha=0.10, label="비타민 대역 (190-255Hz)")
    ax.axhspan(135, 190, color='#3B82F6', alpha=0.10, label="내레이션 대역 (135-190Hz)")
    ax.axhspan(95, 135, color='#A855F7', alpha=0.10, label="ASMR 대역 (95-135Hz)")
    ax.axhspan(50, 95, color='#8B5CF6', alpha=0.10, label="동굴저음 대역 (< 95Hz)")

    for bound in [95, 135, 190, 255]:
        ax.axhline(bound, color='#CBD5E1', linestyle=':', linewidth=1.2)

    f0_valid = np.copy(f0)
    f0_valid[~valid_mask] = np.nan

    ax.plot(times, f0_valid, color='#0284C7', linewidth=2.8, label='음성 피치 궤적 (F0)')
    ax.scatter(times[valid_mask], f0[valid_mask], color='#0369A1', s=14, alpha=0.8)
    ax.axhline(bti_res['mean_f0'], color='#D97706', linestyle='--', linewidth=1.8, label=f"평균치 ({bti_res['mean_f0']}Hz)")

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(60, 360)
    ax.set_xlabel("시간 (초)", color='#334155', fontsize=10.5, fontweight='bold', fontfamily=KOREAN_FONT)
    ax.set_ylabel("피치 주파수 (Hz)", color='#334155', fontsize=10.5, fontweight='bold', fontfamily=KOREAN_FONT)
    ax.set_title("시간별 유효 발화 피치 궤적 (Continuous Pitch Timeline)", color='#0F172A', fontsize=11.5, fontweight='bold', fontfamily=KOREAN_FONT, pad=10)
    ax.tick_params(colors='#334155', labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.grid(color='#E2E8F0', linestyle='--', alpha=0.9)
    leg = ax.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=8.5, ncol=2)
    for text in leg.get_texts():
        text.set_fontfamily(KOREAN_FONT)
    plt.tight_layout()
    return fig


def plot_denoise_spectrogram(y_org, y_clean, sr):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True)
    fig.patch.set_facecolor('#FFFFFF')
    for ax in axes:
        ax.set_facecolor('#F8FAFC')
        for spine in ax.spines.values():
            spine.set_color('#CBD5E1')
        ax.tick_params(colors='#334155')

    stft_org = librosa.amplitude_to_db(np.abs(custom_stft(y_org)), ref=np.max)
    stft_clean = librosa.amplitude_to_db(np.abs(custom_stft(y_clean)), ref=np.max)

    img1 = librosa.display.specshow(stft_org, sr=sr, hop_length=512, x_axis='time', y_axis='linear', ax=axes[0], cmap='viridis')
    axes[0].set_title("입력 신호 원본 스펙트로그램 (Raw STFT)", fontsize=10.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    axes[0].set_ylim(0, 4500)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(stft_clean, sr=sr, hop_length=512, x_axis='time', y_axis='linear', ax=axes[1], cmap='viridis')
    axes[1].set_title("스펙트럼 차감 정제 스펙트로그램 (Clean Denoised)", fontsize=10.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    axes[1].set_ylim(0, 4500)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_algorithm_deep_dive(intermediates):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5))
    fig.patch.set_facecolor('#FFFFFF')

    sr = intermediates['sr']
    n_fft = intermediates['n_fft']
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    freq_mask = freqs <= 3500
    f_disp = freqs[freq_mask]

    # 패널 1: 노이즈 프로파일
    ax1.set_facecolor('#F8FAFC')
    noise_profile = intermediates['noise_profile'][freq_mask]
    ax1.fill_between(f_disp, 0, noise_profile, color='#EF4444', alpha=0.25, label='추정 잡음 스펙트럼 N(k)')
    ax1.plot(f_disp, noise_profile, color='#DC2626', linewidth=2)
    alpha = intermediates['alpha']
    ax1.plot(f_disp, alpha * noise_profile, color='#D97706', linestyle='--', linewidth=1.8, label=f'과차감 기준선 (α={alpha} · N(k))')

    ax1.set_title("패널 A: FFT 기반 주파수별 노이즈 프로파일 N(k)", fontsize=10.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    ax1.set_xlabel("주파수 (Hz)", color='#334155', fontsize=9.5, fontweight='bold', fontfamily=KOREAN_FONT)
    ax1.set_ylabel("진폭 크기", color='#334155', fontsize=9.5, fontweight='bold', fontfamily=KOREAN_FONT)
    ax1.tick_params(colors='#334155')
    ax1.grid(color='#E2E8F0', linestyle='--', alpha=0.9)
    leg1 = ax1.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=8.5)
    for text in leg1.get_texts():
        text.set_fontfamily(KOREAN_FONT)
    for spine in ax1.spines.values():
        spine.set_color('#CBD5E1')

    # 패널 2: 차감 전후 비교
    ax2.set_facecolor('#F8FAFC')
    mid_idx = intermediates['mid_frame']
    org_slice = intermediates['magnitude_original'][freq_mask, mid_idx]
    clean_slice = intermediates['clean_magnitude'][freq_mask, mid_idx]

    ax2.plot(f_disp, org_slice, color='#059669', linewidth=1.8, alpha=0.85, label='원신호 |X(m,k)|')
    ax2.plot(f_disp, alpha * noise_profile, color='#D97706', linestyle=':', linewidth=1.6, label='잡음 임계선 αN(k)')
    ax2.plot(f_disp, clean_slice, color='#0284C7', linewidth=2.4, label='차감 후 |S_clean(m,k)|')

    ax2.set_title(f"패널 B: 프레임 #{mid_idx} 스펙트럼 차감 결과", fontsize=10.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    ax2.set_xlabel("주파수 (Hz)", color='#334155', fontsize=9.5, fontweight='bold', fontfamily=KOREAN_FONT)
    ax2.set_ylabel("진폭 크기", color='#334155', fontsize=9.5, fontweight='bold', fontfamily=KOREAN_FONT)
    ax2.tick_params(colors='#334155')
    ax2.grid(color='#E2E8F0', linestyle='--', alpha=0.9)
    leg2 = ax2.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=8.5)
    for text in leg2.get_texts():
        text.set_fontfamily(KOREAN_FONT)
    for spine in ax2.spines.values():
        spine.set_color('#CBD5E1')
    plt.tight_layout()
    return fig


# =========================================================
# [5단계] Streamlit 웹 애플리케이션 진입점 & UI
# =========================================================
st.set_page_config(
    page_title="목BTI 결과 리포트",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    /* PC에서도 모바일 비율을 완벽히 유지하도록 컨테이너 최대 폭 460px 고정 및 중앙 정렬 */
    .stApp {
        background-color: #F1F5F9 !important;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    .block-container {
        max-width: 460px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 1.5rem 14px 3rem 14px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.08) !important;
        min-height: 100vh !important;
    }

    @media (max-width: 500px) {
        .block-container {
            max-width: 100% !important;
            box-shadow: none !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
        }
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1E293B !important;
    }

    /* 메인 헤더 타이틀 */
    .main-title {
        text-align: center;
        font-size: 23px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        text-align: center;
        font-size: 12.5px;
        color: #64748B;
        margin-bottom: 18px;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Overlapping Character & Result Card (design.md 완벽 준수) */
    .card-wrapper {
        position: relative;
        margin-top: 70px;
        margin-bottom: 22px;
    }

    .result-card {
        position: relative;
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 20px;
        padding: 75px 20px 24px 20px; /* 상단 이미지 공간 + 좌우 균등 */
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        text-align: center;
    }

    .character-img {
        position: absolute;
        top: -65px;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 120px;
        object-fit: contain;
        z-index: 10;
        filter: drop-shadow(0px 8px 14px rgba(0, 0, 0, 0.18));
    }

    .type-badge {
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 8px;
    }

    .result-type-title {
        margin: 0;
        font-size: 26px;
        font-weight: 900;
        color: #0F172A;
        letter-spacing: -0.5px;
    }

    .result-type-en {
        font-size: 12.5px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .result-type-tagline {
        font-weight: 800;
        font-size: 13.5px;
        margin-bottom: 8px;
        line-height: 1.4;
    }

    .result-type-desc {
        font-size: 12px;
        color: #334155;
        line-height: 1.6;
        margin-bottom: 12px;
        font-weight: 500;
    }

    .result-signature {
        background: #F1F5F9;
        border-left: 3px solid #0284C7;
        padding: 10px 12px;
        border-radius: 0 8px 8px 0;
        font-size: 11.5px;
        color: #0F172A;
        font-weight: 600;
        text-align: left;
    }

    /* 2x2 메트릭 그리드 */
    .metrics-grid-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-bottom: 20px;
    }

    .metric-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    }

    .metric-label {
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.3px;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 18px;
        font-weight: 900;
        color: #0F172A;
    }

    /* 패널 카드 */
    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    }

    .panel-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #E2E8F0;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #475569 !important;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 8px 10px;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0284C7 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    button[kind="secondary"], button[kind="primary"] {
        min-height: 42px !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.markdown("### DSP 엔진 설정")
with st.sidebar.expander("노이즈 제거 파라미터", expanded=False):
    method = st.selectbox("잡음 추정 모드", ["Static (초반 정적 구간)", "Dynamic (최소 통계법)"])
    alpha = st.slider("과차감 계수 (α)", 0.5, 4.0, 1.5, 0.1)
    beta = st.slider("스펙트럼 바닥 계수 (β)", 0.001, 0.10, 0.02, 0.005, format="%.3f")
    if "Static" in method:
        noise_duration = st.slider("정적 분석 구간 (초)", 0.1, 2.0, 0.5, 0.1)
        window_duration, bias_factor = 1.5, 1.5
    else:
        window_duration = st.slider("최소 통계 윈도우 (초)", 0.5, 3.0, 1.5, 0.1)
        bias_factor = st.slider("통계 편향 보정치", 1.0, 3.0, 1.5, 0.1)
        noise_duration = 0.5

with st.sidebar.expander("목BTI 주파수 대역 기준 (Hz)", expanded=False):
    th_high = st.number_input("하이톤형 하한선", value=255.0, step=5.0)
    th_vitamin = st.number_input("비타민형 하한선", value=190.0, step=5.0)
    th_narration = st.number_input("내레이션형 하한선", value=135.0, step=5.0)
    th_asmr = st.number_input("ASMR형 하한선", value=95.0, step=5.0)

# 메인 타이틀
st.markdown('<div class="main-title">🎙️ 목BTI 결과 리포트</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">당신의 목소리 주파수를 FFT로 분석해 목소리 성향을 찾아드립니다.</div>', unsafe_allow_html=True)

tab_file, tab_mic, tab_preset = st.tabs(["MP3 파일 업로드", "실시간 마이크", "시뮬레이션"])
audio_bytes_data = None
audio_source_label = ""

with tab_file:
    st.markdown("""
        <div class="panel-card">
            <div class="panel-title">MP3 오디오 파일 업로드</div>
            <p style="color:#475569; font-size:0.88rem; margin-bottom:12px; font-weight:500;">
                목소리가 담긴 <b>MP3</b> (또는 WAV, M4A) 파일을 업로드하세요.
            </p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("MP3 등 오디오 파일 선택 또는 드래그 앤 드롭", type=["mp3", "wav", "m4a", "ogg", "flac", "aac"], key="main_audio_uploader")
    if uploaded_file is not None:
        raw_val = uploaded_file.getvalue()
        if len(raw_val) > 0:
            audio_bytes_data = raw_val
            audio_source_label = uploaded_file.name

with tab_mic:
    st.markdown("""
        <div class="panel-card">
            <div class="panel-title">🎤 실시간 마이크 녹음</div>
            <p style="color:#475569; font-size:0.88rem; margin-bottom:12px; font-weight:500;">
                아래 버튼을 눌러 3~5초간 목소리를 녹음하고, <b>녹음 완료</b> 버튼으로 분석하세요.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if hasattr(st, "audio_input"):
        # Streamlit >= 1.31
        mic_audio = st.audio_input("마이크 녹음 시작", key="official_audio_recorder")
        if mic_audio is not None and audio_bytes_data is None:
            raw_val = mic_audio.getvalue()
            if len(raw_val) > 0:
                audio_bytes_data = raw_val
                audio_source_label = "마이크 실시간 발화"
    else:
        # HTML5 MediaRecorder 기반 녹음 UI
        import streamlit.components.v1 as components

        mic_html = """
        <style>
        .mic-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 4px 0;
        }
        .mic-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            border-radius: 14px;
            border: none;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            width: 100%;
            justify-content: center;
            margin-bottom: 8px;
        }
        #startBtn  { background: #0284C7; color: #fff; }
        #startBtn:hover  { background: #0369A1; }
        #stopBtn   { background: #DC2626; color: #fff; display:none; }
        #stopBtn:hover   { background: #B91C1C; }
        .rec-indicator {
            display: none;
            align-items: center;
            gap: 8px;
            color: #DC2626;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .rec-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #DC2626;
            animation: blink 1s infinite;
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        #statusMsg {
            font-size: 13px;
            color: #475569;
            font-weight: 500;
            margin-bottom: 6px;
            min-height: 20px;
        }
        #audioPreview { width:100%; margin-top:8px; display:none; border-radius:12px; }
        #downloadArea { margin-top:10px; display:none; }
        #downloadBtn {
            background:#059669; color:#fff;
            border:none; border-radius:10px;
            padding:10px 20px; font-size:14px; font-weight:700;
            cursor:pointer; width:100%;
        }
        #downloadBtn:hover { background:#047857; }
        </style>

        <div class="mic-container">
            <div class="rec-indicator" id="recIndicator">
                <div class="rec-dot"></div> 녹음 중...
            </div>
            <div id="statusMsg">🎙️ 버튼을 눌러 녹음을 시작하세요.</div>
            <button class="mic-btn" id="startBtn" onclick="startRec()">🎙️ 녹음 시작</button>
            <button class="mic-btn" id="stopBtn"  onclick="stopRec()">⏹️ 녹음 완료</button>
            <audio id="audioPreview" controls></audio>
            <div id="downloadArea">
                <a id="downloadBtn" style="display:block;text-align:center;padding:10px 20px;background:#059669;color:#fff;border-radius:10px;font-weight:700;font-size:14px;text-decoration:none;">
                    ⬇️ 녹음 파일 다운로드 (분석용 업로드 탭에서 올리세요)
                </a>
                <p style="font-size:11.5px;color:#94A3B8;margin-top:6px;text-align:center;">
                    📌 다운로드 후 <b>[MP3 파일 업로드]</b> 탭에서 업로드하면 바로 분석됩니다.
                </p>
            </div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/lamejs/1.2.1/lame.min.js"></script>
        <script>
        let mediaRecorder, chunks = [], stream;

        async function startRec() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                chunks = [];
                mediaRecorder.ondataavailable = e => { if(e.data.size > 0) chunks.push(e.data); };
                mediaRecorder.onstop = buildAudio;
                mediaRecorder.start(100);

                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'flex';
                document.getElementById('recIndicator').style.display = 'flex';
                document.getElementById('statusMsg').textContent = '🔴 녹음 중 — 말씀하신 후 [녹음 완료]를 누르세요.';
                document.getElementById('audioPreview').style.display = 'none';
                document.getElementById('downloadArea').style.display = 'none';
            } catch(err) {
                document.getElementById('statusMsg').textContent = '❌ 마이크 접근 실패: ' + err.message;
            }
        }

        function stopRec() {
            if(mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
            if(stream) stream.getTracks().forEach(t => t.stop());
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('recIndicator').style.display = 'none';
            document.getElementById('startBtn').style.display = 'flex';
            document.getElementById('statusMsg').textContent = '✅ 녹음 완료! 아래에서 미리 듣고 다운로드하세요.';
        }

        function buildAudio() {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            const url  = URL.createObjectURL(blob);

            // WebM -> AudioBuffer -> MP3 via lamejs
            blob.arrayBuffer().then(function(arrayBuf) {
                const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
                tmpCtx.decodeAudioData(arrayBuf, function(audioBuffer) {
                    const sr = audioBuffer.sampleRate;
                    const raw = audioBuffer.getChannelData(0); // mono
                    const samples = new Int16Array(raw.length);
                    for (let i = 0; i < raw.length; i++) {
                        const s = Math.max(-1, Math.min(1, raw[i]));
                        samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    const mp3enc = new lamejs.Mp3Encoder(1, sr, 128);
                    const blockSize = 1152;
                    const mp3Data = [];
                    for (let i = 0; i < samples.length; i += blockSize) {
                        const chunk = samples.subarray(i, i + blockSize);
                        const encoded = mp3enc.encodeBuffer(chunk);
                        if (encoded.length > 0) mp3Data.push(new Int8Array(encoded));
                    }
                    const flushed = mp3enc.flush();
                    if (flushed.length > 0) mp3Data.push(new Int8Array(flushed));

                    const mp3Blob = new Blob(mp3Data, { type: 'audio/mp3' });
                    const mp3Url = URL.createObjectURL(mp3Blob);

                    const preview = document.getElementById('audioPreview');
                    preview.src = mp3Url;
                    preview.style.display = 'block';

                    const dl = document.getElementById('downloadBtn');
                    dl.href = mp3Url;
                    dl.download = 'mic_recording.mp3';
                    document.getElementById('downloadArea').style.display = 'block';
                }, function() {
                    // fallback: 디코딩 실패 시 WebM 그대로 제공
                    const preview = document.getElementById('audioPreview');
                    preview.src = url;
                    preview.style.display = 'block';
                    const dl = document.getElementById('downloadBtn');
                    dl.href = url;
                    dl.download = 'mic_recording.webm';
                    document.getElementById('downloadArea').style.display = 'block';
                });
            });
        }
        </script>
        """
        components.html(mic_html, height=340)
        st.caption("📌 녹음 완료 후 파일을 다운로드하고, [MP3 파일 업로드] 탭에서 올리면 바로 분석됩니다.")

    st.caption("※ PC/스마트폰 브라우저에서 마이크 권한을 허용해야 작동합니다.")

with tab_preset:
    st.markdown("""
        <div class="panel-card">
            <div class="panel-title">음성 시뮬레이션 프리셋</div>
            <p style="color:#475569; font-size:0.88rem; margin-bottom:12px; font-weight:500;">
                버튼 하나로 5대 대역별 배음 신호를 즉각 시뮬레이션합니다.
            </p>
        </div>
    """, unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    selected_sample = None
    with col_p1:
        if st.button("하이톤 (280Hz)"): selected_sample, audio_source_label = 285.0, "하이톤형 시뮬레이션"
        if st.button("내레이션 (160Hz)"): selected_sample, audio_source_label = 160.0, "내레이션형 시뮬레이션"
        if st.button("동굴저음 (80Hz)"): selected_sample, audio_source_label = 80.0, "동굴저음형 시뮬레이션"
    with col_p2:
        if st.button("비타민 (220Hz)"): selected_sample, audio_source_label = 220.0, "비타민형 시뮬레이션"
        if st.button("ASMR (115Hz)"): selected_sample, audio_source_label = 115.0, "ASMR형 시뮬레이션"

    if selected_sample is not None:
        sr_synth = 22050
        duration = 3.5
        t = np.linspace(0, duration, int(sr_synth * duration), endpoint=False)
        f0_synth = selected_sample + 8.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(f0_synth) / sr_synth
        synth_wave = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase) + 0.04 * np.random.normal(0, 0.05, len(t))
        buf = io.BytesIO()
        sf.write(buf, synth_wave, sr_synth, format='WAV')
        buf.seek(0)
        audio_bytes_data = buf.getvalue()

    if selected_sample is not None:
        sr_synth = 22050
        duration = 3.5
        t = np.linspace(0, duration, int(sr_synth * duration), endpoint=False)
        f0_synth = selected_sample + 8.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(f0_synth) / sr_synth
        synth_wave = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase) + 0.04 * np.random.normal(0, 0.05, len(t))
        buf = io.BytesIO()
        sf.write(buf, synth_wave, sr_synth, format='WAV')
        buf.seek(0)
        audio_bytes_data = buf.getvalue()

# 분석 파이프라인
if audio_bytes_data is not None and len(audio_bytes_data) > 0:
    with st.spinner("오디오 신호 로드 및 정규화 중..."):
        try:
            y_org, sr = decode_audio_bytes(audio_bytes_data)
        except Exception as e:
            st.error(f"오디오 데이터를 로드하는 중 오류가 발생했습니다: {e}")
            st.stop()

    duration_sec = len(y_org) / sr
    if duration_sec < 0.5:
        st.warning("오디오 길이가 너무 짧습니다. 2초 이상 말씀해 주세요.")
        st.stop()

    org_wav_io = io.BytesIO()
    sf.write(org_wav_io, y_org, sr, format='WAV')
    org_wav_io.seek(0)

    with st.spinner("고효율 라이브러리 FFT 기반 스펙트럼 분석 및 노이즈 캔슬링 연산 중..."):
        if "Static" in method:
            y_clean, intermediates = spectral_subtraction_detailed(y_org, sr, noise_duration=noise_duration, alpha=alpha, beta=beta)
        else:
            y_clean, intermediates = spectral_subtraction_minimum_statistics_detailed(y_org, sr, window_duration=window_duration, alpha=alpha, beta=beta, bias_factor=bias_factor)

    max_clean = np.max(np.abs(y_clean))
    if max_clean > 0:
        y_clean = y_clean / max_clean * 0.95

    denoised_io = io.BytesIO()
    sf.write(denoised_io, y_clean, sr, format='WAV')
    denoised_io.seek(0)

    with st.spinner("기본주파수(F0) 추적 및 5대 유형 분석 중..."):
        f0, valid_mask, times, valid_pitches = extract_pitch_and_voicing(y_clean, sr)
        bti_res = classify_voice_bti(valid_pitches, th_high=th_high, th_vitamin=th_vitamin, th_narration=th_narration, th_asmr=th_asmr)

    if not bti_res["success"]:
        st.warning(bti_res['message'])
        st.stop()

    top_type_key = bti_res["top_type"]
    profile = VOICE_BTI_PROFILES[top_type_key]
    char_img_uri = get_image_base64(profile.get("image_file", "dolphin.png"))

    st.markdown("---")
    # 1. Overlapping Character & Result Card (design.md 정확 반영)
    st.markdown(f"""
        <div class="card-wrapper">
            <img src="{char_img_uri}" class="character-img" alt="{profile['name']}">
            <div class="result-card" style="border-left: 8px solid {profile['color']};">
                <div class="type-badge" style="background: {profile['color']};">
                    대표 분류 유형: Top 1 ({bti_res['percentages'][top_type_key]}%)
                </div>
                <h1 class="result-type-title">{profile['name']}</h1>
                <div class="result-type-en">({profile['en_name']})</div>
                <p class="result-type-tagline" style="color: {profile['color']};">{profile['tagline']}</p>
                <p class="result-type-desc">{profile['summary']}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    clean_note = bti_res['note_mean'].replace('♯', '#')
    sub_label = f"{bti_res['sub_type']} ({bti_res['percentages'][bti_res['sub_type']]}%)" if bti_res['sub_type'] else "단일 집중형"

    # 2. 2x2 Metric Grid (모바일 및 PC 일관 비율)
    st.markdown(f"""
        <div class="metrics-grid-container">
            <div class="metric-tile">
                <div class="metric-label">평균 기본주파수 (F0)</div>
                <div class="metric-value" style="color:#0284C7;">{bti_res['mean_f0']} <span style="font-size:0.85rem; color:#64748B;">Hz</span></div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">음악적 대응 음고</div>
                <div class="metric-value" style="color:#059669;">{clean_note}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">유효 발화 음역 폭</div>
                <div class="metric-value" style="color:#D97706; font-size:1.35rem;">{bti_res['min_f0']}~{bti_res['max_f0']} <span style="font-size:0.8rem; color:#64748B;">Hz</span></div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">서브 보이스 성향</div>
                <div class="metric-value" style="color:#7C3AED; font-size:1.15rem;">{sub_label}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. 5대 목BTI 유형별 점유율 및 원 그래프 (Donut Chart)
    st.markdown("### 📊 5대 목BTI 유형별 점유율")
    fig_donut = plot_voice_bti_donut(bti_res)
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    # 5대 유형 랭킹 카드 리스트 (모바일 최적화)
    card_htmls = []
    for idx, t_name in enumerate(ORDERED_TYPES):
        pct = bti_res["percentages"][t_name]
        p_info = VOICE_BTI_PROFILES[t_name]
        is_top = (t_name == top_type_key)
        border_style = f"border: 2px solid {p_info['color']}; box-shadow: 0 4px 12px rgba(0,0,0,0.06);" if is_top else "border: 1px solid #E2E8F0;"
        top_badge = f'<span style="background:{p_info["color"]}; color:#FFF; font-size:0.75rem; font-weight:800; padding:2px 8px; border-radius:4px; margin-left:6px;">BEST</span>' if is_top else ''
        icon_uri = get_image_base64(p_info.get("image_file", ""))
        img_tag = f'<img src="{icon_uri}" style="width:34px; height:34px; object-fit:contain; margin-right:10px;">' if icon_uri else ''
        card_htmls.append(f"""
            <div style="background:#FFFFFF; {border_style} border-radius:14px; padding:12px 16px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center;">
                    {img_tag}
                    <div>
                        <div style="font-weight:800; font-size:0.98rem; color:#0F172A; margin-bottom:2px;">{t_name} {top_badge}</div>
                        <div style="color:#64748B; font-size:0.78rem; font-weight:600;">{p_info['en_name']}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.55rem; font-weight:900; color:{p_info['color']};">{pct}%</div>
                    <div style="background:#F1F5F9; border-radius:999px; height:6px; width:80px; overflow:hidden; margin-top:4px;">
                        <div style="background:{p_info['color']}; width:{pct}%; height:100%; border-radius:999px;"></div>
                    </div>
                </div>
            </div>
        """)
    st.markdown("".join(card_htmls), unsafe_allow_html=True)

    # 4. 주파수 스펙트럼 내 목소리 영역 분포 (모바일 최적화 일체형 카드)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 주파수 스펙트럼 내 목소리 영역 분포")
    clean_note_display = bti_res['note_mean'].replace('♯', '#')
    st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:16px 14px 10px 14px; box-shadow:0 4px 14px rgba(0,0,0,0.04); margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
                <span style="font-size:0.86rem; font-weight:800; color:#334155;">내 발화 음역대</span>
                <div style="display:flex; gap:6px;">
                    <span style="background:#EFF6FF; color:#1D4ED8; font-size:0.78rem; font-weight:700; padding:3px 8px; border-radius:6px; border:1px solid #BFDBFE;">
                        평균 {bti_res['mean_f0']:.1f}Hz ({clean_note_display})
                    </span>
                    <span style="background:#FEF2F2; color:#B91C1C; font-size:0.78rem; font-weight:700; padding:3px 8px; border-radius:6px; border:1px solid #FECACA;">
                        {bti_res['min_f0']}~{bti_res['max_f0']}Hz
                    </span>
                </div>
            </div>
    """, unsafe_allow_html=True)
    fig_spec = plot_voice_spectrum_bar(valid_pitches, bti_res)
    st.plotly_chart(fig_spec, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. 상세 프로파일 리포트 (모바일 수직 스택 카드)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 📋 {profile['name']} 상세 프로파일 리포트")
    st.markdown(f"""
        <div class="panel-card">
            <h4 style="color:{profile['color']}; margin-top:0; font-size:1.05rem;">✨ 음색 및 매력 지표</h4>
            <p style="color:#334155; font-size:0.9rem; line-height:1.6; margin-bottom:10px;">{profile['charm_points']}</p>
            <div style="font-weight:800; color:#0F172A; font-size:0.88rem; margin-bottom:6px;">특징 요소:</div>
            <ul style="color:#475569; font-size:0.86rem; padding-left:18px; line-height:1.6; margin:0;">
                {''.join([f"<li>{trait}</li>" for trait in profile['traits']])}
            </ul>
        </div>
        <div class="panel-card">
            <h4 style="color:#059669; margin-top:0; font-size:1.05rem;">💼 권장 활용 직무 및 역할</h4>
            <p style="color:#334155; font-size:0.9rem; line-height:1.6; margin-bottom:10px;">해당 보이스의 주파수 대역이 최고의 전달력을 발휘하는 분야:</p>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">
                {''.join([f'<span style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; padding:5px 12px; border-radius:6px; font-size:0.82rem; font-weight:700;">{career}</span>' for career in profile['best_careers']])}
            </div>
        </div>
        <div class="panel-card">
            <h4 style="color:#D97706; margin-top:0; font-size:1.05rem;">🤝 최적의 듀오 보이스 조화</h4>
            <div style="background:#FEF3C7; border:1px solid #FCD34D; border-radius:10px; padding:12px; color:#92400E; font-weight:700; font-size:0.9rem; line-height:1.5;">
                {profile['duo_match']}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 6. 노이즈 캔슬링 전/후 오디오 비교 청취
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎧 노이즈 캔슬링 전/후 비교 청취")
    st.markdown("##### 원본 오디오 (Raw Input)")
    st.audio(org_wav_io, format="audio/wav")
    st.caption(f"신호원: {audio_source_label} | {sr} Hz | {duration_sec:.2f}초")

    st.markdown("##### 정제 오디오 (FFT Denoised)")
    st.audio(denoised_io, format="audio/wav")
    st.download_button(label="정제 음원 다운로드 (WAV)", data=denoised_io, file_name="denoised_voice.wav", mime="audio/wav")

    # 7. 수학적 모델링 및 DSP 심층 분석 (뒤쪽에 몰아서 배치)
    st.markdown("---")
    with st.expander("🔬 DSP 오디오 심층 분석 및 수학적 모델링 (상세)", expanded=False):
        st.markdown("#### 1. 시간별 유효 발화 주파수 궤적 (Continuous Pitch Contour)")
        fig_contour = plot_pitch_contour(times, f0, valid_mask, bti_res)
        st.pyplot(fig_contour)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 2. 2D 주파수 스펙트로그램 비교 (STFT Decibel Scale)")
        fig_stft = plot_denoise_spectrogram(y_org, y_clean, sr)
        st.pyplot(fig_stft)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 3. 핵심 알고리즘 원리 및 수학적 모델링")

        st.markdown("##### (1) 고효율 고속 푸리에 변환 (Fast Fourier Transform, scipy.fft)")
        st.markdown("PocketFFT 알고리즘 기반 고속 라이브러리를 활용하여 시간 영역의 오디오를 $O(N \\log_2 N)$의 최고 속도로 주파수 영역으로 변환합니다.")
        st.latex(r"X(k) = \sum_{n=0}^{N-1} x(n) e^{-j \frac{2\pi}{N} k n}, \quad k = 0, \dots, N-1")

        st.markdown("##### (2) 스펙트럼 차감법 (Spectral Subtraction)")
        st.markdown("추정된 잡음 스펙트럼 $\\hat{N}(k)$를 과차감 계수 $\\alpha$와 스펙트럼 바닥 $\\beta$를 통해 감산합니다:")
        st.latex(r"|\hat{S}(m, k)| = \max\Big( |X(m, k)| - \alpha \cdot \hat{N}(k),\quad \beta \cdot \hat{N}(k) \Big)")
        st.latex(r"\hat{S}_{\text{complex}}(m, k) = |\hat{S}(m, k)| \cdot e^{j \angle X(m, k)} \quad \xrightarrow{\text{ISTFT (OLA)}} \quad \hat{s}(t)")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### (3) 알고리즘 단계별 연산 시각화 (Noise Profile & Spectral Subtraction)")
        fig_algo = plot_algorithm_deep_dive(intermediates)
        st.pyplot(fig_algo)

else:
    st.info("상단 탭에서 [MP3 파일 업로드]를 통해 음성을 분석하시거나, [시뮬레이션] 탭의 버튼을 클릭해 보세요.")
