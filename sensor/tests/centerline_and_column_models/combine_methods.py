import numpy as np
import cv2
from scipy.signal import savgol_filter, find_peaks
from analyze_six import distro_based_filament_extraction, preprocess_image
from analyze_five import detect_filament_global_sigma
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import savgol_filter, find_peaks


def detect_filament_split(
    img_bgr,
    laser_y,
    window_radius=20,
    min_segment_len=8,
    gap_close=4,
    sigma_frac_thresh=0.06,
    red_ratio_thresh=1.15,
    center_prior_hfhw=50,
    min_fit_width=6,
    max_fit_width=None,
    skew_threshold=0,
    skew_full=0.2,
    a1_threshold=0,
    a1_full=.4,
    min_split_confidence=0.55,
    debug=False,
):
    """
    Compact filament + split detector.

    Main meanings:
      confidence[x]       = per-column confidence that filament exists.
      column_evidence[x]  = combined per-column evidence used to locate the split range.
      split_confidence    = scalar confidence that a detected filament segment is split.

    Requires your existing:
      - distro_based_filament_extraction
      - detect_filament_global_sigma
    """

    H, W = img_bgr.shape[:2]

    def safe_savgol(v, window_length=17, polyorder=3, deriv=0):
        v = np.asarray(v, dtype=np.float32)
        n = len(v)

        if n < 5:
            return np.gradient(v) if deriv else v.copy()

        win = min(window_length, n if n % 2 else n - 1)
        if win < 3:
            return np.gradient(v) if deriv else v.copy()

        if win % 2 == 0:
            win -= 1

        poly = min(polyorder, win - 1)

        return savgol_filter(
            v,
            window_length=win,
            polyorder=poly,
            deriv=deriv,
            mode="interp",
        )

    def robust01(v, lo_q=5, hi_q=95):
        v = np.asarray(v, dtype=np.float32)
        lo, hi = np.nanpercentile(v, [lo_q, hi_q])
        return np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    def compute_skew_confidence():
        red = img_bgr[:, :, 2].astype(np.float32)

        y0 = max(0, int(laser_y - window_radius))
        y1 = min(H, int(laser_y + window_radius + 1))
        ys = np.arange(y0, y1, dtype=np.float32)

        confidence = np.zeros(W, dtype=np.float32)
        skews = np.zeros(W, dtype=np.float32)
        skew_mu = np.zeros(W, dtype=np.float32)
        skew_sigma = np.zeros(W, dtype=np.float32)
        skew_energy = np.zeros(W, dtype=np.float32)

        for x in range(W):
            profile = red[y0:y1, x].astype(np.float32)

            # Small background subtraction. This is only to reduce shadow influence.
            profile = profile - np.percentile(profile, 20)
            profile = np.clip(profile, 0, None)

            total = profile.sum()
            skew_energy[x] = total

            if total < 1:
                continue

            mu = np.sum(profile * ys) / total
            var = np.sum(profile * (ys - mu) ** 2) / total

            if var < 1e-6:
                continue

            sigma = np.sqrt(var)

            skew = (
                np.sum(profile * (ys - mu) ** 3) / total
            ) / (sigma ** 3)

            skew_mu[x] = mu
            skew_sigma[x] = sigma
            skews[x] = skew

            # This remains filament confidence, not split confidence.
            # skew_mag = abs(skew)
            skew_mag = skew
            confidence[x] = np.clip(
                (skew_mag - skew_threshold) / (skew_full - skew_threshold + 1e-6),
                0.0,
                1.0,
            )

        confidence = safe_savgol(confidence, 9, 2)
        skews = safe_savgol(skews, 9, 2)

        return confidence, skews, {
            "skew_mu": skew_mu,
            "skew_sigma": skew_sigma,
            "skew_energy": skew_energy,
            "skew_y0": y0,
            "skew_y1": y1,
        }

    def compute_two_peak_score():
        red = img_bgr[:, :, 2].astype(np.float32)

        y0 = max(0, int(laser_y - window_radius))
        y1 = min(H, int(laser_y + window_radius + 1))

        score = np.zeros(W, dtype=np.float32)
        return np.ones(W, dtype=np.float32)
        for x in range(W):
            p = red[y0:y1, x].astype(np.float32)
            p = p - np.percentile(p, 20)
            p = np.clip(p, 0, None)

            mx = float(np.max(p))
            if mx < 5:
                continue

            peaks, _ = find_peaks(
                p,
                distance=3,
                prominence=max(1.0, 0.06 * mx),
            )

            if len(peaks) < 2:
                continue

            top = peaks[np.argsort(p[peaks])[-2:]]
            top = np.sort(top)

            h1 = float(p[top[0]])
            h2 = float(p[top[1]])

            peak_hi = max(h1, h2)
            peak_lo = min(h1, h2)

            ratio = peak_lo / (peak_hi + 1e-6)
            sep = float(top[1] - top[0])
            valley = float(np.min(p[top[0]:top[1] + 1]))
            valley_depth = 1.0 - valley / (peak_hi + 1e-6)

            ratio_score = np.clip((ratio - 0.25) / 0.75, 0, 1)
            sep_score = np.clip((sep - 3) / max(window_radius * 0.5, 1), 0, 1)
            valley_score = np.clip((valley_depth - 0.15) / 0.65, 0, 1)

            score[x] = np.clip(
                0.35 * ratio_score +
                0.30 * sep_score +
                0.35 * valley_score,
                0,
                1,
            )

        return safe_savgol(score, 7, 2)

    # ------------------------------------------------------------------
    # 1. Keep the earlier filament extraction flow.
    # ------------------------------------------------------------------
    gaussian_pack, _ = distro_based_filament_extraction(img_bgr)
    A, gaussian_means, sigmas = gaussian_pack

    A = np.asarray(A, dtype=np.float32)
    gaussian_means = np.asarray(gaussian_means, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)

    rolling_win = min(101, W if W % 2 else W - 1)
    rolling_win = max(rolling_win, 9)
    if rolling_win % 2 == 0:
        rolling_win += 1

    segments, labels, features, sigma_mask, diagnostics = detect_filament_global_sigma(
        img_bgr[:, :, ::-1],
        A,
        gaussian_means,
        sigmas,
        use_detrend=False,
        detrend_method="rolling",
        rolling_win=rolling_win,
        sigma_frac_thresh=sigma_frac_thresh,
        min_segment_len=min_segment_len,
        gap_close=gap_close,
        red_ratio_thresh=red_ratio_thresh,
        debug=False,
    )

    # ------------------------------------------------------------------
    # 2. Keep all column evidence signals.
    # ------------------------------------------------------------------
    confidence, skews, skew_diag = compute_skew_confidence()

    skews_smooth = safe_savgol(skews, 17, 3)
    skew_grad = safe_savgol(skews_smooth, 15, 3, deriv=1)

    two_peak = compute_two_peak_score()

    sigma_med = float(np.nanmedian(sigmas)) + 1e-6
    sigma_lift = np.clip((sigmas / sigma_med - 1.0) / 0.35, 0.0, 1.0)

    amp_score = robust01(A, 10, 90)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y0 = max(0, int(laser_y - window_radius))
    y1 = min(H, int(laser_y + window_radius + 1))
    local_brightness = gray[y0:y1].mean(axis=0)
    brightness_score = robust01(local_brightness, 5, 85)

    column_evidence = (
        0.30 * confidence +
        0.25 * sigma_lift +
        0.20 * two_peak +
        0.15 * amp_score +
        0.10 * brightness_score
    )

    # This is still per-column evidence, not split confidence.
    column_evidence *= np.clip(0.35 + 0.65 * brightness_score, 0.0, 1.0)
    column_evidence = safe_savgol(column_evidence, 11, 2)

    # Used only for locating a good cubic-fit interval.
    grad_for_search = skew_grad * (column_evidence ** 2)

    # ------------------------------------------------------------------
    # 3. Compute scalar split confidence per detected filament segment.
    # ------------------------------------------------------------------
    candidates = []

    xs_full = np.arange(W, dtype=np.float32)
    center = 0.5 * (W - 1)
    sigma_prior = center_prior_hfhw / np.sqrt(2.0 * np.log(2.0))

    start_prior = np.exp(
        -0.5 * ((xs_full - center) / (sigma_prior + 1e-6)) ** 2
    )

    for i, (lo, hi) in enumerate(segments):
        lo = int(lo)
        hi = int(hi)

        if hi - lo + 1 < min_fit_width + 4:
            continue

        # --------------------------------------------------------------
        # Find split start.
        # Center prior is used ONLY here.
        # --------------------------------------------------------------
        start_max = hi - min_fit_width
        if start_max <= lo:
            continue

        valid_start = np.zeros(W, dtype=bool)
        valid_start[lo:start_max + 1] = True

        start_score = np.full(W, -np.inf, dtype=np.float32)
        start_score[valid_start] = (
            grad_for_search[valid_start] *
            start_prior[valid_start]
        )

        split_start = int(np.argmax(start_score))

        if not np.isfinite(start_score[split_start]):
            continue

        # --------------------------------------------------------------
        # Find split end.
        # No center prior here. End can be anywhere right of start.
        # --------------------------------------------------------------
        end_lo = split_start + min_fit_width
        end_hi = hi

        if max_fit_width is not None:
            end_hi = min(end_hi, split_start + max_fit_width)

        if end_lo >= end_hi:
            continue

        end_region = skew_grad[end_lo:end_hi + 1]

        if len(end_region) == 0:
            continue

        split_end = int(end_lo + np.argmin(end_region))

        if split_end <= split_start:
            continue

        # --------------------------------------------------------------
        # Cubic fit.
        # This is the actual split-confidence source.
        # --------------------------------------------------------------
        fit_region = (grad_for_search)[split_start:split_end + 1].astype(np.float32)

        if len(fit_region) < 4:
            continue

        x = np.linspace(-1.0, 1.0, len(fit_region)).astype(np.float32)
        y = fit_region - np.mean(fit_region)
        
        # Preserves your original scaling style.
        y = y * 256.0

        M = np.column_stack([
            np.ones_like(x),
            x,
            x ** 2,
            x ** 3,
        ])

        a0, a1, a2, a3 = np.linalg.lstsq(M, y, rcond=None)[0]

        fit = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3

        # Main split signal: positive linear coefficient.
        positive_a1 = max(float(a1), 0.0)

        linear_score = np.clip(
            (positive_a1- a1_threshold) / (a1_full - a1_threshold + 1e-6),
            0.0,
            1.0,
        )

        # Diagnostic shape score similar to your original a1 / abs(a3).
        linear_over_cubic = positive_a1 / (abs(float(a3)) + 1e-6)
        ratio_score = np.clip(linear_over_cubic / 0.5, 0.0, 1.0)

        # Fit quality gate.
        ss_res = float(np.sum((y - fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-6
        r2 = 1.0 - ss_res / ss_tot
        r2_score = np.clip((r2 - 0.10) / 0.70, 0.0, 1.0)

        # Filament support gate inside the fit interval.
        inside = slice(split_start, split_end + 1)

        filament_support = float(np.percentile(confidence[inside], 75))
        evidence_support = float(np.max(column_evidence[inside]))
        two_peak_support = float(np.percentile(two_peak[inside], 75))
        sigma_support = float(np.mean(sigma_lift[inside]))
        brightness_support = float(np.mean(brightness_score[inside]))

        support_score = np.clip(
            0.45 * filament_support +
            0.25 * evidence_support +
            0.15 * two_peak_support +
            0.15 * sigma_support,
            0.0,
            1.0,
        )

        # Start should be positive, end should be negative.
        start_strength = max(float(grad_for_search[split_start]), 0.0)
        end_strength = max(float(-grad_for_search[split_end]), 0.0)

        edge_balance = min(start_strength, end_strength) / (
            max(start_strength, end_strength) + 1e-6
        )
        edge_balance = float(np.clip(edge_balance, 0.0, 1.0))

        # Dark-region penalty.
        dark_gate = np.clip((brightness_support - 0.15) / 0.60, 0.0, 1.0)

        # Final scalar split confidence for this filament segment.
        # a1 remains the dominant term.
        split_confidence = linear_score
        split_confidence *= 0.70 + 0.30 * ratio_score
        split_confidence *= 0.65 + 0.35 * r2_score
        split_confidence *= 0.60 + 0.40 * support_score
        split_confidence *= 0.65 + 0.35 * edge_balance
        split_confidence *= dark_gate

        split_confidence = float(np.clip(split_confidence, 0.0, 1.0))

        f = features[i] if i < len(features) else {}
        label = labels[i] if i < len(labels) else "unknown"
        
        candidate = {
            "segment_index": int(i),
            "filament_start": int(lo),
            "filament_end": int(hi),

            "split_start": int(split_start),
            "split_end": int(split_end),
            "split_center": float(0.5 * (split_start + split_end)),

            # Scalar confidence that this filament segment is split.
            "split_confidence": split_confidence,

            # Keep this alias only if old plotting code expects "confidence".
            # It means split confidence here, not per-column filament confidence.
            #"confidence": split_confidence,

            "is_split": bool(split_confidence >= min_split_confidence),
            "label": label,
            "features": f,

            "coefficients": {
                "a0": float(a0),
                "a1": float(a1),
                "a2": float(a2),
                "a3": float(a3),
            },

            "linear_score": float(linear_score),
            "linear_over_cubic": float(linear_over_cubic),
            "ratio_score": float(ratio_score),
            "r2": float(r2),
            "r2_score": float(r2_score),
            "support_score": float(support_score),
            "filament_support": float(filament_support),
            "evidence_support": float(evidence_support),
            "two_peak_support": float(two_peak_support),
            "sigma_support": float(sigma_support),
            "brightness_support": float(brightness_support),
            "edge_balance": float(edge_balance),
            "dark_gate": float(dark_gate),

            "fit_x": x,
            "fit_y": y,
            "fit": fit,
        }

        candidates.append(candidate)

    candidates.sort(key=lambda d: d["support_score"], reverse=True)

    best_split = candidates[0] if candidates else None
    is_split = bool(
        best_split is not None and
        best_split["split_confidence"] >= min_split_confidence
    )

    if debug:
        print("segments:", segments)
        print("labels:", labels)
        print("best_split:", best_split)

    return {
        "is_split": is_split,
        "best_split": best_split,
        "candidates": candidates,
        "segments": segments,
        "labels": labels,
        "features": features,
        "diagnostics": diagnostics,
        "signals": {
            "A": A,
            "gaussian_means": gaussian_means,
            "sigmas": sigmas,
            # Per-column filament confidence.
            "confidence": confidence,
            "skews": skews,
            "skew_grad": skew_grad,
            # Per-column combined evidence.
            "column_evidence": column_evidence,
            "grad_for_search": grad_for_search,
            "weighted_grad": grad_for_search,
            "two_peak": two_peak,
            "sigma_lift": sigma_lift,
            "amp_score": amp_score,
            "brightness_score": brightness_score,
            **skew_diag,
        },
 }



def plot_filament_split_result(
    img_bgr,
    result,
    save_path=None,
    show=True,
    top_n_candidates=5,
    title="Filament split compact detector debug",
):
    """
    Matplotlib debug plot for detect_filament_split_compact().

    Plots:
      1. Image with detected filament segments and split ranges
      2. Per-column filament confidence and column evidence
      3. Individual evidence channels
      4. Skew, skew gradient, and search gradient
      5. Gaussian fit signals
      6. Candidate split confidences
      7. Cubic fit for the best split candidate

    Parameters
    ----------
    img_bgr : np.ndarray
        Original BGR image.

    result : dict
        Return value from detect_filament_split_compact().

    save_path : str or None
        If not None, saves figure to this path.

    show : bool
        If True, calls plt.show().

    top_n_candidates : int
        Number of candidates to annotate/plot in candidate summary.

    title : str
        Figure title.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """

    signals = result.get("signals", {})
    segments = result.get("segments", [])
    candidates = result.get("candidates", [])
    best = result.get("best_split", None)

    H, W = img_bgr.shape[:2]
    xs = np.arange(W)

    def get_signal(name, default=0.0):
        v = signals.get(name, None)
        if v is None:
            return np.full(W, default, dtype=np.float32)
        return np.asarray(v, dtype=np.float32)

    def norm(v):
        v = np.asarray(v, dtype=np.float32)
        if len(v) == 0:
            return v
        lo, hi = np.nanpercentile(v, [2, 98])
        return np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    confidence = get_signal("confidence")
    column_evidence = get_signal("column_evidence")
    two_peak = get_signal("two_peak")
    sigma_lift = get_signal("sigma_lift")
    amp_score = get_signal("amp_score")
    brightness_score = get_signal("brightness_score")

    skews = get_signal("skews")
    skew_grad = get_signal("skew_grad")
    grad_for_search = get_signal("grad_for_search")

    A = get_signal("A")
    gaussian_means = get_signal("gaussian_means")
    sigmas = get_signal("sigmas")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 15),
        gridspec_kw={
            "height_ratios": [3.0, 1.3, 1.5, 1.5]#, 1.4, 1.2, 1.8]
        },
    )

    # fig.suptitle(title, fontsize=14)

    def shade_segments(ax, alpha=0.12):
        for lo, hi in segments:
            ax.axvspan(lo, hi, alpha=alpha)

    def draw_candidate_lines(ax, include_all=True):
        if include_all:
            for c in candidates[:top_n_candidates]:
                if "split_start" in c:
                    ax.axvline(c["split_start"], linestyle="--", alpha=0.25)
                if "split_end" in c:
                    ax.axvline(c["split_end"], linestyle="--", alpha=0.25)

        if best is not None:
            ax.axvline(best["split_start"], linewidth=2.5)
            ax.axvline(best["split_end"], linewidth=2.5)
            ax.axvline(best["split_center"], linestyle=":", linewidth=2.5)

    # ------------------------------------------------------------------
    # 1. Image overlay
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.imshow(img_rgb)
    #ax.set_title("Image overlay: filament segments and split fit ranges")
    ax.set_ylabel("y")

    shade_segments(ax, alpha=0.18)
    draw_candidate_lines(ax)

    for i, (lo, hi) in enumerate(segments):
        ax.text(
            lo,
            4,
            f"seg {i}",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
        )

    if best is not None:
        coeff = best.get("coefficients", {})
        label = (
            f"best split_conf={best['split_confidence']:.3f} | "
            f"a1={coeff.get('a1', np.nan):.3f} | "
            f"range=({best['split_start']}, {best['split_end']})"
        )
    else:
        label = "no split candidate"

    ax.text(
        0.01,
        0.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # ------------------------------------------------------------------
    # 2. Main per-column signals
    # ------------------------------------------------------------------
    ax = axes[1]
    shade_segments(ax)
    ax.plot(xs, confidence, label="filament confidence")
    ax.plot(xs, column_evidence, label="column evidence", linewidth=2.2)
    draw_candidate_lines(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Filament confidence vs combined column evidence")
    ax.set_ylabel("score")
    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # 3. Evidence channels
    # ------------------------------------------------------------------
    ax = axes[2]
    # shade_segments(ax)
    # ax.plot(xs, two_peak, label="two_peak")
    # ax.plot(xs, sigma_lift, label="sigma_lift")
    # ax.plot(xs, amp_score, label="amp_score")
    # ax.plot(xs, brightness_score, label="brightness_score")
    # draw_candidate_lines(ax)
    # ax.set_ylim(-0.05, 1.05)
    # ax.set_title("Individual evidence channels")
    # ax.set_ylabel("score")
    # ax.legend(loc="upper right")
    # # ------------------------------------------------------------------
    # # 4. Skew and gradients
    # # ------------------------------------------------------------------
    # ax = axes[3]
    shade_segments(ax)

    def norm_with_params(v):
        v = np.asarray(v, dtype=np.float32)
        lo, hi = np.nanpercentile(v, [2, 98])
        v_norm = np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        return v_norm, lo, hi

    def apply_norm(v, lo, hi):
        v = np.asarray(v, dtype=np.float32)
        return np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    skews_norm, _, _ = norm_with_params(skews)
    skew_grad_norm, _, _ = norm_with_params(skew_grad)
    grad_search_norm, grad_lo, grad_hi = norm_with_params(grad_for_search)

    ax.plot(xs, skews_norm, label="skews normalized")
    ax.plot(xs, skew_grad_norm, label="skew_grad normalized")
    ax.plot(xs, grad_search_norm, label="grad_for_search normalized")

    # --------------------------------------------------------------
    # Add cubic fit back onto the x-column graph.
    # The fit was made on:
    #   y = (grad_for_search[start:end+1] - mean) * 128
    #
    # So to plot it on the same axis as grad_for_search:
    #   fit_raw = fit / 128 + mean
    # Then normalize using the SAME grad_for_search normalization.
    # --------------------------------------------------------------
    if best is not None:
        split_start = int(best["split_start"])
        split_end = int(best["split_end"])

        fit = np.asarray(best.get("fit", []), dtype=np.float32)

        if len(fit) > 0 and split_end > split_start:
            region_x = np.linspace(split_start, split_end, len(fit))

            raw_region = grad_for_search[split_start:split_end + 1]
            raw_mean = float(np.mean(raw_region))

            fit_raw = fit / 128.0 + raw_mean
            fit_norm = apply_norm(fit_raw, grad_lo, grad_hi)

            ax.plot(
                region_x,
                fit_norm,
                linewidth=3.0,
                label="cubic fit on grad_for_search",
            )

    draw_candidate_lines(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Skew, skew gradient, search gradient, and cubic fit")
    ax.set_ylabel("normalized")
    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # 5. Gaussian signals
    # ------------------------------------------------------------------
    ax = axes[3]
    # shade_segments(ax)
    # ax.plot(xs, norm(A), label="A normalized")
    # ax.plot(xs, norm(sigmas), label="sigma normalized")
    # ax.plot(xs, norm(gaussian_means), label="gaussian mean normalized")
    # draw_candidate_lines(ax)
    # ax.set_ylim(-0.05, 1.05)
    # ax.set_title("Gaussian fit signals")
    # ax.set_ylabel("normalized")
    # ax.legend(loc="upper right")

    # # ------------------------------------------------------------------
    # # 6. Candidate scalar split confidences
    # # ------------------------------------------------------------------
    # ax = axes[5]
    shade_segments(ax)

    if len(candidates) > 0:
        plot_candidates = candidates[:top_n_candidates]

        centers = [c["split_center"] for c in plot_candidates]
        split_confs = [c["split_confidence"] for c in plot_candidates]

        ax.scatter(centers, split_confs, s=55, label="candidate split_confidence")

        for c in plot_candidates:
            ax.text(
                c["split_center"],
                c["split_confidence"] + 0.035,
                f"{c['split_confidence']:.2f}",
                ha="center",
                fontsize=8,
            )

    if best is not None:
        ax.scatter(
            [best["split_center"]],
            [best["split_confidence"]],
            s=130,
            marker="x",
            label="best candidate",
        )

    draw_candidate_lines(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Scalar split confidence per candidate")
    ax.set_ylabel("split confidence")
    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # 7. Cubic fit for best candidate
    # ------------------------------------------------------------------
    # ax = axes[6]

    # if best is not None:
    #     fit_x = np.asarray(best.get("fit_x", []), dtype=np.float32)
    #     fit_y = np.asarray(best.get("fit_y", []), dtype=np.float32)
    #     fit = np.asarray(best.get("fit", []), dtype=np.float32)

    #     if len(fit_x) > 0 and len(fit_y) > 0:
    #         ax.plot(fit_x, fit_y, label="fit_y")
    #     if len(fit_x) > 0 and len(fit) > 0:
    #         ax.plot(fit_x, fit, label="cubic fit", linewidth=2.2)

    #     coeff = best.get("coefficients", {})
    #     fit_title = (
    #         "Best candidate cubic fit: "
    #         f"a1={coeff.get('a1', np.nan):.3f}, "
    #         f"a3={coeff.get('a3', np.nan):.3f}, "
    #         f"R²={best.get('r2', np.nan):.3f}, "
    #         f"split_conf={best.get('split_confidence', np.nan):.3f}"
    #     )
    #     ax.set_title(fit_title)
    #     ax.set_xlabel("normalized x inside split range")
    #     ax.set_ylabel("gradient fit value")
    #     ax.legend(loc="upper right")
    # else:
    #     ax.text(
    #         0.5,
    #         0.5,
    #         "No split candidate",
    #         ha="center",
    #         va="center",
    #         transform=ax.transAxes,
    #     )
    #     ax.set_title("Best candidate cubic fit")
    #     ax.set_xlabel("normalized x inside split range")

    for ax in axes[:3]:
        ax.set_xlim(0, W - 1)
        ax.set_xlabel("x column")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
def number(p):
    match = re.search(r'\d+', p)
    return int(match.group()) if match else -1
def plot_path(path,label=0):
    img_bgr = preprocess_image(path )
    result = detect_filament_split(img_bgr, laser_y=42, window_radius=40, debug=True)
    fig, ax = plot_filament_split_result(img_bgr, result, show=True, title=f"{label}")
if __name__ == "__main__":
    import random
    import re
    import os
    from analyze_six import max_deviation_from_centerline
    folder = r"C:\Users\dhruv\Documents\dhruv_python\frames\\"
    photos = sorted(os.listdir(folder), key = number)
    #random.shuffle(photos)
    allsigs = []
    for p in photos[285:330]:
        INPUT_PATH = folder + p
        img_bgr = preprocess_image(INPUT_PATH)
        result = detect_filament_split(img_bgr, laser_y=42, window_radius=40, debug=True)
        fig, ax = plot_filament_split_result(img_bgr, result, show=False)
        max_dev = max_deviation_from_centerline(
            INPUT_PATH,
            med_ksize=11,
            search_radius=4,
            smooth_method='savgol',
            savgol_window=7,
            visualize=True
        )