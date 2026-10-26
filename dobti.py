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
import subprocess
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from scipy.ndimage import minimum_filter1d
import imageio_ffmpeg
import streamlit as st

# =========================================================
# [1단계] 한글 폰트(맑은 고딕) 자동 등록 (폰트 깨짐 100% 방지)
# =========================================================
font_path = "C:/Windows/Fonts/malgun.ttf"
try:
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    KOREAN_FONT = font_prop.get_name()
    plt.rcParams['font.family'] = KOREAN_FONT
except Exception:
    KOREAN_FONT = 'Malgun Gothic'
    plt.rcParams['font.family'] = KOREAN_FONT

plt.rcParams['font.sans-serif'] = [KOREAN_FONT, 'NanumGothic', 'Gulim', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# FFmpeg 경로 시스템 연동
try:
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass


# =========================================================
# [2단계] Radix-2 Cooley-Tukey FFT & STFT 알고리즘 직접 구현
# =========================================================
def fft_cooley_tukey(x):
    """Vectorized Radix-2 Cooley-Tukey FFT 알고리즘 (N은 2의 거듭제곱)"""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N <= 1:
        return x
    if (N & (N - 1)) != 0:
        raise ValueError(f"길이 N={N}은 Radix-2 FFT를 위해 2의 거듭제곱이어야 합니다.")

    even = fft_cooley_tukey(x[0::2])
    odd = fft_cooley_tukey(x[1::2])

    if x.ndim == 1:
        k = np.arange(N // 2)
        factor = np.exp(-2j * np.pi * k / N)
        return np.concatenate([even + factor * odd, even - factor * odd])
    else:
        k = np.arange(N // 2)[:, None]
        factor = np.exp(-2j * np.pi * k / N)
        return np.vstack([even + factor * odd, even - factor * odd])


def ifft_cooley_tukey(X):
    """켤레 복소수 쌍대성을 이용한 Inverse Radix-2 Cooley-Tukey FFT"""
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    return np.conj(fft_cooley_tukey(np.conj(X))) / N


def custom_stft(y, n_fft=2048, hop_length=512, window='hann'):
    """Cooley-Tukey FFT 기반 단시간 푸리에 변환 (STFT)"""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    n_fft_pow2 = 1 << (n_fft - 1).bit_length()
    if n_fft_pow2 != n_fft:
        n_fft = n_fft_pow2

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
    spectrum_full = fft_cooley_tukey(windowed_frames)
    return spectrum_full[:n_fft // 2 + 1, :]


def custom_istft(S, hop_length=512, length=None, window='hann'):
    """Cooley-Tukey IFFT 및 Overlap-Add(OLA)를 이용한 시간 영역 음성 신호 복원"""
    n_freq, num_frames = S.shape
    n_fft = (n_freq - 1) * 2

    neg_spectrum = np.conj(S[n_freq - 2:0:-1, :])
    S_full = np.vstack([S, neg_spectrum])

    time_frames = np.real(ifft_cooley_tukey(S_full))
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
# [3단계] 오디오 디코딩, 노이즈 캔슬링 & 목BTI 분류기
# =========================================================
VOICE_BTI_PROFILES = {
    "하이톤형": {
        "id": "high_tone", "name": "하이톤형", "en_name": "High-Tone Sparkling", "color": "#00F2FE",
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
        "tagline": "호감도 급상승! 친근하고 산뜻한 생기발랄 보이스",
        "summary": "자연스러운 밝음과 생기가 묻어나는 상큼한 중고음 보이스입니다. 누구에게나 다정하고 친근하게 다가가는 독보적인 친화력을 자랑합니다.",
        "traits": ["적절한 억양과 산뜻한 리듬감으로 지루할 틈이 없는 스피치", "듣는 이에게 기분 좋은 온기와 긍정적 영감을 불어넣음", "어떤 대화에서도 어색함을 없애는 마법 같은 친화력"],
        "charm_points": "첫마디만 들어도 무장해제되는 친절함과 밝은 미소가 연상되는 음색",
        "best_careers": ["피플팀(HR) 리크루터", "브랜드 마케터", "고객 경험(CX) 전략가", "방송 리포터", "프로덕트 매니저(PM)"],
        "duo_match": "ASMR형 (활기와 차분함이 만나 균형 잡힌 최고의 편안함 선사)",
        "signature_phrase": "“안녕하세요 여러분! 오늘 하루도 기분 좋게 출발해 볼까요?”"
    },
    "내레이션형": {
        "id": "narration", "name": "내레이션형", "en_name": "Narration Classic", "color": "#3B82F6",
        "tagline": "신뢰도 200%! 귀에 꽂히는 또렷하고 안정감 있는 정석 톤",
        "summary": "차분하고 지적이며 전달력이 매우 우수한 표준적이고 균형 잡힌 보이스입니다. 사람들에게 확신과 안정감을 심어주며 프레젠테이션과 설득에 강합니다.",
        "traits": ["불필요한 떨림 없이 곧고 명확하게 꽂히는 발음과 톤", "정보 전달력이 뛰어나 듣는 이가 오래 들어도 피로하지 않음", "지적이고 프로페셔널한 인상을 주는 안정적인 피치"],
        "charm_points": "단번에 사람을 사로잡는 신뢰감과 믿음직스러운 어른의 분위기",
        "best_careers": ["뉴스 앵커 / 아나운서", "다큐멘터리 내레이터", "스타트업 대표 / IR 전문가", "전문 컨설턴트", "변호사 / 법조인"],
        "duo_match": "비타민형 (신뢰감에 발랄함을 더해주는 환상의 비즈니스 파트너)",
        "signature_phrase": "“지금부터 전해드릴 소식에 주목해 주시기 바랍니다.”"
    },
    "ASMR형": {
        "id": "asmr", "name": "ASMR형", "en_name": "ASMR Cozy", "color": "#A855F7",
        "tagline": "귓가를 감싸는 포근함! 마음을 녹여주는 나직하고 감미로운 힐링 톤",
        "summary": "부드럽고 나직하게 속삭이듯 다가와 지친 하루의 피로를 풀어주는 힐링 보이스입니다. 따뜻한 공명과 숨결이 어우러져 듣는 이에게 심리적 안정감을 선물합니다.",
        "traits": ["자극 없이 귓가에 스며드는 따뜻하고 둥근 톤", "숨소리와 잔잔한 울림이 편안하게 조화를 이룸", "새벽 감성을 자극하는 포근하고 몽환적인 감미로움"],
        "charm_points": "잠들기 전 침대 맡에서 듣고 싶은 고막 힐링과 다정한 속삭임",
        "best_careers": ["오디오북 전문 낭독가", "심리 상담사 / 멘탈 코치", "사운드 디자이너", "심야 라디오 진행자", "웰니스 디렉터"],
        "duo_match": "내레이션형 (지적인 설명과 감성적 위로가 어우러지는 완벽한 콤비)",
        "signature_phrase": "“오늘 하루도 정말 고생 많았어요. 편안한 밤 보내세요.”"
    },
    "동굴저음형": {
        "id": "cave_bass", "name": "동굴저음형", "en_name": "Cave Deep Bass", "color": "#8B5CF6",
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
    except Exception as e:
        raise ValueError(f"오디오 형식을 디코딩할 수 없습니다: {e}")


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
    """pYIN 알고리즘을 통한 F0 기본주파수 연속 궤적 추출"""
    max_amp = np.max(np.abs(y))
    y_norm = y / max_amp if max_amp > 1e-6 else y

    rms = librosa.feature.rms(y=y_norm, frame_length=2048, hop_length=hop_length)[0]
    non_zero_rms = rms[rms > 1e-4]
    rms_threshold = max(1e-4, np.percentile(non_zero_rms, 5) * 0.5) if len(non_zero_rms) > 0 else 1e-4

    f0, voiced_flag, _ = librosa.pyin(
        y_norm, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop_length, fill_na=np.nan
    )
    valid_mask = voiced_flag & (~np.isnan(f0)) & (rms[:len(f0)] >= rms_threshold)
    if np.sum(valid_mask) < 5:
        valid_mask = voiced_flag & (~np.isnan(f0))

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
# [4단계] Matplotlib 고대비 라이트 모드 차트 (폰트 깨짐 없음)
# =========================================================
def plot_voice_spectrum_bar(valid_pitches, bti_res):
    fig, ax = plt.subplots(figsize=(16, 4.4))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    x_min, x_max = 60.0, 360.0
    th = bti_res["thresholds"]
    zones = [
        ("동굴저음형\n(< 95Hz)", x_min, th["th_asmr"], "#8B5CF6", 0.16),
        ("ASMR형\n(95 - 135Hz)", th["th_asmr"], th["th_narration"], "#A855F7", 0.18),
        ("내레이션형\n(135 - 190Hz)", th["th_narration"], th["th_vitamin"], "#3B82F6", 0.20),
        ("비타민형\n(190 - 255Hz)", th["th_vitamin"], th["th_high"], "#10B981", 0.20),
        ("하이톤형\n(> 255Hz)", th["th_high"], x_max, "#06B6D4", 0.18),
    ]

    for name, z_start, z_end, color, alpha in zones:
        ax.axvspan(z_start, z_end, ymin=0.15, ymax=0.85, color=color, alpha=alpha)
        ax.axvline(z_start, color='#94A3B8', linestyle='--', linewidth=1.2)
        mid = (z_start + z_end) / 2
        ax.text(mid, 0.50, name, color='#0F172A', fontsize=12, fontweight='bold', fontfamily=KOREAN_FONT,
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.2))

    ax.axvline(x_max, color='#94A3B8', linestyle='--', linewidth=1.2)

    min_f0 = max(x_min, bti_res["min_f0"])
    max_f0 = min(x_max, bti_res["max_f0"])
    if max_f0 - min_f0 < 8.0:
        max_f0 = min(x_max, min_f0 + 8.0)
    mean_f0 = bti_res["mean_f0"]

    rect = patches.FancyBboxPatch(
        (min_f0, 0.18), max_f0 - min_f0, 0.64,
        boxstyle="round,pad=0.03", linewidth=2.8, edgecolor='#E11D48', facecolor='#FDA4AF', alpha=0.55, zorder=4
    )
    ax.add_patch(rect)

    note_clean = bti_res['note_mean'].replace('♯', '#')
    ax.plot([mean_f0, mean_f0], [0.05, 0.95], color='#D97706', linewidth=4, zorder=5)
    ax.scatter([mean_f0], [0.50], color='#D97706', s=240, zorder=6, edgecolor='#FFFFFF', linewidth=3)
    ax.text(mean_f0, 1.01, f"평균 기본주파수: {mean_f0:.1f}Hz ({note_clean})",
            color='#B45309', fontsize=13, fontweight='bold', fontfamily=KOREAN_FONT, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=1.8))

    ax.text(min_f0, 0.05, f"최저 {min_f0:.0f}Hz", color='#BE123C', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT, ha='center', va='top')
    ax.text(max_f0, 0.05, f"최고 {max_f0:.0f}Hz", color='#BE123C', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT, ha='center', va='top')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.25)
    ax.set_yticks([])
    ticks = [60, 95, 135, 190, 255, 300, 360]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}Hz" for t in ticks], fontsize=11.5, color='#1E293B', fontweight='bold', fontfamily=KOREAN_FONT)
    ax.set_xlabel("주파수 대역 (Frequency Hz)", color='#334155', fontsize=12, labelpad=10, fontweight='bold', fontfamily=KOREAN_FONT)
    ax.tick_params(colors='#64748B', bottom=True, length=6)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    plt.tight_layout()
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
    ax.set_xlabel("시간 (초)", color='#334155', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax.set_ylabel("피치 주파수 (Hz)", color='#334155', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax.set_title("시간별 기본주파수 연속 궤적 (Continuous Pitch Contour Timeline)", color='#0F172A', fontsize=13, fontweight='bold', fontfamily=KOREAN_FONT, pad=12)
    ax.tick_params(colors='#334155', labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.grid(color='#E2E8F0', linestyle='--', alpha=0.9)
    leg = ax.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5, ncol=3)
    for text in leg.get_texts():
        text.set_fontfamily(KOREAN_FONT)
    plt.tight_layout()
    return fig


def plot_denoise_spectrogram(y_org, y_clean, sr):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.8), sharey=True)
    fig.patch.set_facecolor('#FFFFFF')
    for ax in axes:
        ax.set_facecolor('#F8FAFC')
        for spine in ax.spines.values():
            spine.set_color('#CBD5E1')
        ax.tick_params(colors='#334155')

    stft_org = librosa.amplitude_to_db(np.abs(custom_stft(y_org)), ref=np.max)
    stft_clean = librosa.amplitude_to_db(np.abs(custom_stft(y_clean)), ref=np.max)

    img1 = librosa.display.specshow(stft_org, sr=sr, hop_length=512, x_axis='time', y_axis='linear', ax=axes[0], cmap='viridis')
    axes[0].set_title("입력 신호 원본 스펙트로그램 (Raw Cooley-Tukey STFT)", fontsize=11.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    axes[0].set_ylim(0, 4500)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(stft_clean, sr=sr, hop_length=512, x_axis='time', y_axis='linear', ax=axes[1], cmap='viridis')
    axes[1].set_title("스펙트럼 차감 정제 스펙트로그램 (Clean Denoised)", fontsize=11.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    axes[1].set_ylim(0, 4500)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_algorithm_deep_dive(intermediates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.4))
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

    ax1.set_title("패널 A: Cooley-Tukey FFT 기반 주파수별 노이즈 프로파일 N(k)", fontsize=11.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    ax1.set_xlabel("주파수 (Hz)", color='#334155', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax1.set_ylabel("스펙트럼 진폭 크기", color='#334155', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax1.tick_params(colors='#334155')
    ax1.grid(color='#E2E8F0', linestyle='--', alpha=0.9)
    leg1 = ax1.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5)
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
    ax2.plot(f_disp, clean_slice, color='#0284C7', linewidth=2.4, label='차감 후 보이스 |S_clean(m,k)|')

    ax2.set_title(f"패널 B: 단일 시간 프레임(Frame #{mid_idx}) 스펙트럼 차감 연산 결과", fontsize=11.5, fontweight='bold', fontfamily=KOREAN_FONT, color='#0F172A')
    ax2.set_xlabel("주파수 (Hz)", color='#334155', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax2.set_ylabel("스펙트럼 진폭 크기", color='#334155', fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax2.tick_params(colors='#334155')
    ax2.grid(color='#E2E8F0', linestyle='--', alpha=0.9)
    leg2 = ax2.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5)
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
    page_title="Voice-BTI Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    .stApp { background-color: #F8FAFC !important; color: #0F172A !important; font-family: 'Pretendard', -apple-system, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
    section[data-testid="stSidebar"] * { color: #1E293B !important; }
    .studio-header { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04); }
    .studio-badge { display: inline-block; font-size: 0.75rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: #0284C7; background: #E0F2FE; border: 1px solid #BAE6FD; padding: 4px 12px; border-radius: 6px; margin-bottom: 10px; }
    .studio-title { font-size: 2.6rem; font-weight: 900; letter-spacing: -1px; color: #0F172A; margin: 0 0 8px 0; }
    .studio-subtitle { font-size: 1.05rem; color: #475569; margin: 0; line-height: 1.6; font-weight: 500; }
    .panel-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03); }
    .panel-title { font-size: 1.2rem; font-weight: 800; color: #0F172A; margin-bottom: 8px; }
    .hero-container { background: #FFFFFF; border: 2px solid #CBD5E1; border-radius: 16px; padding: 32px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05); }
    .hero-tag { font-size: 0.85rem; font-weight: 800; padding: 6px 14px; border-radius: 6px; color: #FFFFFF; display: inline-block; margin-bottom: 14px; }
    .hero-type-title { font-size: 2.8rem; font-weight: 900; color: #0F172A; letter-spacing: -0.5px; margin-bottom: 6px; }
    .hero-tagline { font-size: 1.25rem; font-weight: 700; color: #0284C7; margin-bottom: 14px; }
    .hero-desc { font-size: 1.05rem; line-height: 1.75; color: #334155; margin-bottom: 18px; font-weight: 500; }
    .hero-signature { background: #F1F5F9; border-left: 4px solid #0284C7; padding: 14px 20px; border-radius: 0 8px 8px 0; font-size: 1rem; color: #0F172A; font-weight: 600; }
    .metric-tile { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02); }
    .metric-label { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.5px; color: #64748B; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { font-size: 1.8rem; font-weight: 900; color: #0F172A; }
    h1, h2, h3, h4, h5 { color: #0F172A !important; font-weight: 800 !important; }
    p, span, label { color: #334155 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #E2E8F0; padding: 6px; border-radius: 10px; border: 1px solid #CBD5E1; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #475569 !important; font-weight: 700; font-size: 0.95rem; padding: 10px 22px; border: none; background: transparent; }
    .stTabs [aria-selected="true"] { background: #FFFFFF !important; color: #0284C7 !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); }
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

# 헤더
st.markdown("""
    <div class="studio-header">
        <span class="studio-badge">DSP AUDIO ANALYSIS ENGINE</span>
        <h1 class="studio-title">Voice-BTI Studio</h1>
        <p class="studio-subtitle">자체 제작 Radix-2 Cooley-Tukey FFT 노이즈 캔슬링 및 기본주파수(F0) 스펙트럼 분석기</p>
    </div>
""", unsafe_allow_html=True)

tab_file, tab_mic, tab_preset = st.tabs(["오디오 파일 분석", "실시간 마이크 녹음", "알고리즘 시뮬레이션 프리셋"])
audio_bytes_data = None
audio_source_label = ""

with tab_file:
    st.markdown("""
        <div class="panel-card">
            <div class="panel-title">오디오 파일 업로드 분석</div>
            <p style="color:#475569; font-size:0.95rem; margin-bottom:14px; font-weight:500;">
                스마트폰 음성 녹음 파일(m4a, mp3, wav 등)이나 오디오 파일을 끌어다 놓으세요.
            </p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("오디오 파일 선택 또는 드래그 앤 드롭", type=["m4a", "wav", "mp3", "ogg", "flac", "aac"], key="main_audio_uploader", label_visibility="collapsed")
    if uploaded_file is not None:
        raw_val = uploaded_file.getvalue()
        if len(raw_val) > 0:
            audio_bytes_data = raw_val
            audio_source_label = uploaded_file.name

with tab_mic:
    st.markdown("""
        <div class="panel-card">
            <div class="panel-title">실시간 마이크 녹음</div>
            <p style="color:#475569; font-size:0.95rem; margin-bottom:14px; font-weight:500;">
                녹음 버튼을 눌러 3초~5초간 말씀하신 후 정지 버튼을 누르시면 실시간으로 분석됩니다.
            </p>
        </div>
    """, unsafe_allow_html=True)
    mic_audio = st.audio_input("마이크 녹음", key="official_audio_recorder", label_visibility="collapsed")
    if mic_audio is not None and audio_bytes_data is None:
        raw_val = mic_audio.getvalue()
        if len(raw_val) > 0:
            audio_bytes_data = raw_val
            audio_source_label = "마이크 실시간 발화"
    st.caption("※ PC에 마이크나 헤드셋(이어폰)이 연결되어 있을 때 정상 작동합니다.")

with tab_preset:
    st.markdown("""
        <div class="panel-card">
            <div class="panel-title">알고리즘 검증용 음성 시뮬레이션 프리셋</div>
            <p style="color:#475569; font-size:0.95rem; margin-bottom:14px; font-weight:500;">
                버튼 하나로 5대 주파수 대역별 배음 신호를 생성하여 즉각 시뮬레이션합니다.
            </p>
        </div>
    """, unsafe_allow_html=True)
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    selected_sample = None
    if col_p1.button("하이톤 (280Hz)", width='stretch'): selected_sample, audio_source_label = 285.0, "하이톤형 시뮬레이션"
    if col_p2.button("비타민 (220Hz)", width='stretch'): selected_sample, audio_source_label = 220.0, "비타민형 시뮬레이션"
    if col_p3.button("내레이션 (160Hz)", width='stretch'): selected_sample, audio_source_label = 160.0, "내레이션형 시뮬레이션"
    if col_p4.button("ASMR (115Hz)", width='stretch'): selected_sample, audio_source_label = 115.0, "ASMR형 시뮬레이션"
    if col_p5.button("동굴저음 (80Hz)", width='stretch'): selected_sample, audio_source_label = 80.0, "동굴저음형 시뮬레이션"

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

    with st.spinner("Cooley-Tukey FFT 노이즈 캔슬링 연산 중..."):
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

    st.markdown("---")
    st.markdown(f"""
        <div class="hero-container" style="border-left: 8px solid {profile['color']};">
            <span class="hero-tag" style="background: {profile['color']};">
                대표 분류 유형: Top 1 ({bti_res['percentages'][top_type_key]}%)
            </span>
            <div class="hero-type-title">{profile['name']} <span style="font-size:1.35rem; color:#64748B; font-weight:600; margin-left:12px;">({profile['en_name']})</span></div>
            <div class="hero-tagline">{profile['tagline']}</div>
            <div class="hero-desc">{profile['summary']}</div>
            <div class="hero-signature"><b>대표 시그니처 발화:</b> {profile['signature_phrase']}</div>
        </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""<div class="metric-tile"><div class="metric-label">평균 기본주파수 (F0)</div><div class="metric-value" style="color:#0284C7;">{bti_res['mean_f0']} <span style="font-size:1rem; color:#64748B;">Hz</span></div></div>""", unsafe_allow_html=True)
    with col_m2:
        clean_note = bti_res['note_mean'].replace('♯', '#')
        st.markdown(f"""<div class="metric-tile"><div class="metric-label">음악적 대응 음고</div><div class="metric-value" style="color:#059669;">{clean_note}</div></div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""<div class="metric-tile"><div class="metric-label">유효 발화 음역 폭</div><div class="metric-value" style="color:#D97706; font-size:1.5rem;">{bti_res['min_f0']} ~ {bti_res['max_f0']} <span style="font-size:0.9rem; color:#64748B;">Hz</span></div></div>""", unsafe_allow_html=True)
    with col_m4:
        sub_label = f"{bti_res['sub_type']} ({bti_res['percentages'][bti_res['sub_type']]}%)" if bti_res['sub_type'] else "단일 집중형"
        st.markdown(f"""<div class="metric-tile"><div class="metric-label">서브 보이스 성향</div><div class="metric-value" style="color:#7C3AED; font-size:1.3rem;">{sub_label}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 주파수 스펙트럼 내 목소리 영역 분포")
    fig_spec = plot_voice_spectrum_bar(valid_pitches, bti_res)
    st.pyplot(fig_spec, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 5대 목BTI 유형별 점유율 (%)")
    pct_cols = st.columns(5)
    for idx, t_name in enumerate(ORDERED_TYPES):
        pct = bti_res["percentages"][t_name]
        p_info = VOICE_BTI_PROFILES[t_name]
        is_top = (t_name == top_type_key)
        border_style = f"border: 2.5px solid {p_info['color']}; box-shadow: 0 4px 14px rgba(0,0,0,0.06);" if is_top else "border: 1px solid #E2E8F0;"
        top_badge = f'<span style="background:{p_info["color"]}; color:#FFF; font-size:0.75rem; font-weight:800; padding:2px 8px; border-radius:4px; margin-left:6px;">BEST</span>' if is_top else ''
        with pct_cols[idx]:
            st.markdown(f"""
                <div style="background:#FFFFFF; {border_style} border-radius:14px; padding:18px 14px; text-align:center;">
                    <div style="font-weight:800; font-size:1.05rem; color:#0F172A; margin-bottom:8px;">{t_name} {top_badge}</div>
                    <div style="font-size:2rem; font-weight:900; color:{p_info['color']}; margin-bottom:8px;">{pct}%</div>
                    <div style="background:#F1F5F9; border-radius:999px; height:8px; overflow:hidden; margin-bottom:8px;">
                        <div style="background:{p_info['color']}; width:{pct}%; height:100%; border-radius:999px;"></div>
                    </div>
                    <div style="color:#64748B; font-size:0.82rem; font-weight:600;">{p_info['en_name']}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 시간별 목소리 주파수 궤적 (Continuous Pitch Contour)")
    fig_contour = plot_pitch_contour(times, f0, valid_mask, bti_res)
    st.pyplot(fig_contour, width='stretch')

    st.markdown("---")
    st.markdown("### 핵심 알고리즘 원리 및 수학적 모델링")
    st.markdown("#### 1. Cooley-Tukey Radix-2 고속 푸리에 변환 (FFT)")
    st.markdown("신호를 짝수(Even)와 홀수(Odd) 인덱스로 재귀 분할하여 연산량을 $O(N^2) \\rightarrow O(N \\log_2 N)$으로 비약적으로 최적화합니다.")
    st.latex(r"X(k) = \sum_{m=0}^{N/2-1} x(2m) e^{-j \frac{2\pi}{N} k(2m)} + \sum_{m=0}^{N/2-1} x(2m+1) e^{-j \frac{2\pi}{N} k(2m+1)}")
    st.latex(r"\begin{cases} X(k) = E(k) + W_N^k \cdot O(k) \\ X\left(k + \frac{N}{2}\right) = E(k) - W_N^k \cdot O(k) \end{cases} \quad \left(W_N = e^{-j \frac{2\pi}{N}}\right)")

    st.markdown("#### 2. 스펙트럼 차감법 (Spectral Subtraction)")
    st.markdown("추정된 잡음 스펙트럼 $\\hat{N}(k)$를 과차감 계수 $\\alpha$와 스펙트럼 바닥 $\\beta$를 통해 감산합니다:")
    st.latex(r"|\hat{S}(m, k)| = \max\Big( |X(m, k)| - \alpha \cdot \hat{N}(k),\quad \beta \cdot \hat{N}(k) \Big)")
    st.latex(r"\hat{S}_{\text{complex}}(m, k) = |\hat{S}(m, k)| \cdot e^{j \angle X(m, k)} \quad \xrightarrow{\text{ISTFT (OLA)}} \quad \hat{s}(t)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 알고리즘 단계별 연산 시각화 (Noise Profile & Spectral Subtraction)")
    fig_algo = plot_algorithm_deep_dive(intermediates)
    st.pyplot(fig_algo, width='stretch')

    st.markdown("---")
    st.markdown(f"### {profile['name']} 상세 프로파일 리포트")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f"""
            <div class="panel-card" style="height:100%;">
                <h4 style="color:{profile['color']}; margin-top:0;">음색 및 매력 지표</h4>
                <p style="color:#334155; font-size:0.95rem; line-height:1.7;">{profile['charm_points']}</p>
                <div style="margin-top:16px; font-weight:800; color:#0F172A; font-size:0.9rem;">특징 요소:</div>
                <ul style="color:#475569; font-size:0.92rem; padding-left:20px; line-height:1.7; margin-top:6px;">
                    {''.join([f"<li>{trait}</li>" for trait in profile['traits']])}
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown(f"""
            <div class="panel-card" style="height:100%;">
                <h4 style="color:#059669; margin-top:0;">권장 활용 직무 및 역할</h4>
                <p style="color:#334155; font-size:0.95rem; line-height:1.7;">해당 보이스의 주파수 대역이 최고의 전달력을 발휘하는 분야입니다:</p>
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:14px;">
                    {''.join([f'<span style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; padding:6px 14px; border-radius:6px; font-size:0.88rem; font-weight:700;">{career}</span>' for career in profile['best_careers']])}
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_c3:
        st.markdown(f"""
            <div class="panel-card" style="height:100%;">
                <h4 style="color:#D97706; margin-top:0;">최적의 듀오 보이스 조화</h4>
                <p style="color:#334155; font-size:0.95rem; line-height:1.7;">주파수 상보적 대조를 이루며 최상의 하모니를 만들어내는 파트너 톤:</p>
                <div style="background:#FEF3C7; border:1px solid #FCD34D; border-radius:10px; padding:16px; margin-top:10px; color:#92400E; font-weight:700; font-size:0.98rem;">{profile['duo_match']}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 노이즈 캔슬링 전/후 비교 청취 및 스펙트로그램")
    col_audio1, col_audio2 = st.columns(2)
    with col_audio1:
        st.markdown("##### 원본 오디오 (Raw Input)")
        st.audio(org_wav_io, format="audio/wav")
        st.caption(f"신호원: {audio_source_label} | 샘플링 주파수: {sr} Hz | 길이: {duration_sec:.2f}초")
    with col_audio2:
        st.markdown("##### 정제 오디오 (Cooley-Tukey Denoised)")
        st.audio(denoised_io, format="audio/wav")
        st.download_button(label="정제 음원 다운로드 (WAV)", data=denoised_io, file_name="denoised_voice.wav", mime="audio/wav")

    with st.expander("2D 주파수 스펙트로그램 비교 상세 (STFT Decibel Scale)", expanded=False):
        fig_stft = plot_denoise_spectrogram(y_org, y_clean, sr)
        st.pyplot(fig_stft, width='stretch')

else:
    st.info("상단 탭에서 [오디오 파일 분석]을 통해 파일을 업로드하시거나, [알고리즘 시뮬레이션 프리셋] 버튼을 클릭해 보세요.")

