import numpy as np
from .timer import timer
from scipy.special import erfc
from scipy.optimize import root
from functools import lru_cache
from math import sqrt

"""
Functions for analyzing and extracting statistics from the 2D eye histograms.
"""


def nrz_histogram_analysis(
    msmts: dict,
    counts: dict,
    hist_data: np.ndarray,
    bathtub: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    dx: float,
    dy: float,
    threshold: float,
    pattern_counts: np.ndarray,
    tdec_s_noise: float,
    tdec_M1: float,
    tdec_M2: float,
    tdec_BER: float,
) -> dict:
    sargs = {
        "hd": np.sum(hist_data, axis=2),
        "bathtub": bathtub,
        "ymin": ymin,
        "ymax": ymax,
        "dx": dx,
        "dy": dy,
        "thr": threshold,
    }
    sargs["sidx"] = round(hist_data.shape[0] / 2)  # sampling index (x)
    f = (threshold - ymin) / (ymax - ymin)
    sargs["tidx"] = round(f * hist_data.shape[1])  # threshold index (y)
    sargs["y"] = np.arange(hist_data.shape[1]) * sargs["dy"] + sargs["ymin"]
    sargs["x"] = np.arange(hist_data.shape[0]) * sargs["dx"]

    """
    Perform standard histogram measurements for a NRZ signal
    """
    msmts["inner_eye_height"] = compute_inner_eye_height(sargs)
    msmts["inner_eye_width"] = compute_inner_eye_width(sargs)
    msmts.update(compute_stats_from_nrz_bathtub(sargs))
    if msmts["eye_height_0.025"] > 0:
        msmts["vecp_xp"] = 10 * np.log10(msmts["oma_xp"] / msmts["eye_height_0.025"])
        msmts["vecp_4140"] = 10 * np.log10(msmts["oma_4140"] / msmts["eye_height_0.025"])
        msmts["vecp_8180"] = 10 * np.log10(msmts["oma_8180"] / msmts["eye_height_0.025"])
    msmts["d_lev"] = compute_d_lev(hist_data, sargs)
    counts["d_lev"] = pattern_counts

    tdec_R = compute_tdec_r(sargs, msmts, tdec_s_noise, tdec_M1, tdec_M2, tdec_BER)
    for oma_type in ["xp", "4140", "8180"]:
        msmts[f"tdec_{oma_type}"] = 10*np.log10((msmts[f"oma_{oma_type}"]/2) * (1 / (Qinv_BER(tdec_BER) * tdec_R)))

@timer
def pam4_histogram_analysis(
    msmts: dict,
    counts: dict,
    hist_data: np.ndarray,
    bathtub: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    dx: float,
    dy: float,
    threshold_lower: float,
    threshold: float,
    threshold_upper: float,
    pattern_counts: np.ndarray,
    tdecq_s_noise: float,
    tdecq_ceq: float,
    tdecq_BER: float,
) -> dict:
    sargs = {
        "hd": np.sum(hist_data, axis=2),
        "bathtub": bathtub,
        "ymin": ymin,
        "ymax": ymax,
        "dx": dx,
        "dy": dy,
        "thr": threshold,
    }
    sargs["sidx"] = round(hist_data.shape[0] / 2)  # sampling index (x)
    f1 = (threshold_lower - ymin) / (ymax - ymin)
    f2 = (threshold - ymin) / (ymax - ymin)
    f3 = (threshold_upper - ymin) / (ymax - ymin)
    sargs["tidx1"] = round(f1 * hist_data.shape[1])  # lower threshold index (y)
    sargs["tidx2"] = round(f2 * hist_data.shape[1])  # middle threshold index (y)
    sargs["tidx3"] = round(f3 * hist_data.shape[1])  # upper threshold index (y)
    sargs["y"] = np.arange(hist_data.shape[1]) * sargs["dy"] + sargs["ymin"]
    sargs["x"] = np.arange(hist_data.shape[0]) * sargs["dx"]

    """
    Perform standard histogram measurements for a PAM4 signal
    """
    msmts.update(compute_stats_from_pam4_bathtub(sargs))
    
    msmts["d_lev"] = compute_d_lev(hist_data, sargs)
    counts["d_lev"] = pattern_counts

    tdec_R = compute_tdecq_r(sargs, msmts, tdecq_s_noise, tdecq_ceq, tdecq_BER)
    Qt = 3.414  # TODO: compute this explicitly.
    msmts["tdecq_outer"] = 10*np.log10((msmts["oma_outer"]/6) * (1 / (Qt * tdec_R)))

@timer
def compute_d_lev(hist_data: np.ndarray, d: dict) -> float:
    return [np.dot(hist_data[d["sidx"], :, zi], d["y"])/np.sum(hist_data[d["sidx"], :, zi]) for zi in range(hist_data.shape[2])]

@timer
def compute_inner_eye_height(d: dict) -> float:
    try:
        upper_vslice = d["hd"][d["sidx"], d["tidx"] :]
        uidx = np.nonzero(upper_vslice)[0][0]
        upper = uidx * d["dy"]
        lower_vslice = d["hd"][d["sidx"], : d["tidx"]]
        lidx = np.nonzero(lower_vslice[::-1])[0][0]
        lower = lidx * d["dy"]
        return upper + lower
    except:
        return np.nan


@timer
def compute_inner_eye_width(d: dict) -> float:
    try:
        upper_hslice = d["hd"][d["sidx"] :, d["tidx"]]
        uidx = np.nonzero(upper_hslice)[0][0]
        upper = uidx * d["dx"]
        lower_hslice = d["hd"][: d["sidx"], d["tidx"]]
        lidx = np.nonzero(lower_hslice[::-1])[0][0]
        lower = lidx * d["dx"]
        return upper + lower
    except IndexError:
        return np.nan


@timer
def compute_stats_from_nrz_bathtub(d: dict) -> dict:
    v = {}
    v["vertical_ber"] = d["bathtub"][d["sidx"], d["tidx"]]
    v["vertical_ber_optimized"] = np.min(d["bathtub"][d["sidx"], :])

    target_bers = [0.025, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    for ber in target_bers:
        mask = d["bathtub"][d["sidx"], :] < ber
        diff = np.diff(mask)  # len(diff) = len(mask) - 1

        if True in diff:
            icenter = np.argmin(d["bathtub"][d["sidx"], :])
            upper_height_idx = icenter
            lower_height_idx = icenter - 1
            if True in diff[icenter:]:
                upper_height_idx += np.where(diff[icenter:] == True)[0][-1]
            if True in diff[:icenter]:
                lower_height_idx -= np.where(np.flip(diff[:icenter]) == True)[0][-1]

            v[f"upper_eye_height_{ber}"] = (0.5 + (upper_height_idx - d["tidx"])) * d["dy"]
            v[f"lower_eye_height_{ber}"] = (0.5 + (d["tidx"] - lower_height_idx)) * d["dy"]
        else:
            v[f"upper_eye_height_{ber}"] = 0
            v[f"lower_eye_height_{ber}"] = 0

        v[f"eye_height_{ber}"] = v[f"lower_eye_height_{ber}"] + v[f"upper_eye_height_{ber}"]

        mask = d["bathtub"][:, d["tidx"]] < ber
        diff = np.diff(mask)  # len(diff) = len(mask) - 1
        if True in diff[d["sidx"] :]:
            upper_width_idx = np.where(diff[d["sidx"] :] == True)[0][0]
            v[f"upper_eye_width_{ber}"] = (0.5 + upper_width_idx) * d["dx"]
        else:
            v[f"upper_eye_width_{ber}"] = 0
        if True in diff[: d["sidx"]]:
            lower_width_idx = np.where(np.flip(diff[: d["sidx"]]) == True)[0][0]
            v[f"lower_eye_width_{ber}"] = (0.5 + lower_width_idx) * d["dx"]
        else:
            v[f"lower_eye_width_{ber}"] = 0

        v[f"lower_eye_width_{ber}"] = 0 if v[f"lower_eye_width_{ber}"] < 0 else v[f"lower_eye_width_{ber}"]
        v[f"upper_eye_width_{ber}"] = 0 if v[f"upper_eye_width_{ber}"] < 0 else v[f"upper_eye_width_{ber}"]
        v[f"eye_width_{ber}"] = v[f"lower_eye_width_{ber}"] + v[f"upper_eye_width_{ber}"]
    return v

@timer
def compute_stats_from_pam4_bathtub(d: dict) -> dict:
    v = {}
    v["vertical_ser"] = sum([d["bathtub"][d["sidx"], d[tidx], i] for i, tidx in enumerate(["tidx1", "tidx2", "tidx3"])])
    return v

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

@lru_cache
def Qinv_BER(BER):
    sol = root(lambda x: np.log10(Q(x)) - np.log10(BER), 0)
    return float(sol.x[0])

@timer
def compute_tdec_r(
    d: dict, 
    msmts: dict, 
    S: float,
    M1: float,  # M1 is the mode partition noise: Eq (95-4)
    M2: float,  # M2 is the modal noise: Eq (95-5)
    TARGET_BER: float,
) -> float:
    """
    Compute the noise that could be added by a receiver, as detailed in Equation (95-3)
    of IEEE Std 802.3-2015 Standards for Ethernet
    """

    # First find the 0 and 1 UI crossing points, determined by the average of the eye
    # diagram crossing times.  This function is only executed if the sampling_offset_mode
    # is set to "half_ui".
    ui = d["x"] / (max(d["x"]) + d["dx"])
    idx = np.arange(len(ui))
    W = 0.04
    Pavg = msmts["average"]

    hist_halves = (d["hd"][:, d["tidx"] :], d["hd"][:, : d["tidx"]])
    y_upper, y_lower = (d["y"][d["tidx"] :], d["y"][: d["tidx"]])
    halves = ("upper", "lower")
    hist = {}
    sigmas = []

    # Grab the 4x vertical histograms, between 0.4 and 0.6 UI (with width of 0.04 UI),
    # and vertically above & below Pavg.
    for side, hp in zip(("left", "right"), (0.4, 0.6)):
        for half, hist_half in zip(halves, hist_halves):
            mask = (ui < (hp+W/2)) & (ui > (hp-W/2))
            idx1, idx2 = idx[1:][np.diff(mask)]
            hist[side+"_"+half] = np.sum(hist_half[idx1:idx2, :], axis=0)

        # Compute the values of sigma such that we satisfy Eq. (95-2) and the 
        # BER is equal to TARGET_BER
        def get_ber(sigma):
            t1 = np.dot(hist[side+"_upper"], Q((y_upper - Pavg) / sigma)) / np.sum(hist[side+"_upper"])
            t2 = np.dot(hist[side+"_lower"], Q((Pavg - y_lower) / sigma)) / np.sum(hist[side+"_lower"])
            return 0.5 * (t1 + t2)

        sigma_guess = (y_upper[np.argmax(hist[side+"_upper"])] - Pavg)
        sol = root(lambda sigma: (np.log10(get_ber(abs(sigma))) - np.log10(TARGET_BER)), x0=sigma_guess, tol=1e-6)
        sigmas.append(abs(sol.x[0]))

    N = min(sigmas) 
    return (1 - M1) * np.sqrt(N**2 + S**2 - M2**2)


@timer
def compute_tdecq_r(
    d: dict, 
    msmts: dict, 
    S: float,
    Ceq: float,
    TARGET_BER: float,
) -> float:
    """
    Compute the noise that could be added by a receiver, as detailed in Equation (121-11)
    of IEEE Std 802.3bs-2017 Standards for Ethernet
    """
    # First find the 0 and 1 UI crossing points, determined by the average of the eye
    # diagram crossing times.  This function is only executed if the sampling_offset_mode
    # is set to "half_ui".
    ui = d["x"] / (max(d["x"]) + d["dx"])
    idx = np.arange(len(ui))
    W = 0.04
    TARGET_SER = TARGET_BER * 2
    Pth1 = msmts["average"] - msmts["oma_outer"]/3
    Pth2 = msmts["average"]
    Pth3 = msmts["average"] + msmts["oma_outer"]/3

    hist = {}
    sigmas = []

    # Grab the 8x vertical histograms, between 0.45 and 0.55 UI (with width of 0.04 UI),
    # and vertically on segments divided by Pth1, Pth2, Pth3
    for side, hp in zip(("left", "right"), (0.45, 0.55)):
        mask = (ui < (hp+W/2)) & (ui > (hp-W/2))
        idx1, idx2 = idx[1:][np.diff(mask)]
        hist["F_"+side] = np.sum(d["hd"][idx1:idx2, :], axis=0)

    # Normalize the histograms, so that for each side the sum of all samples is = 1.
    # 802.3 defines the two "sides", the left and right as F_L and F_R.
    for side in ("left", "right"):
        hist["F_"+side] /= np.sum(hist["F_"+side])

    dy = abs(d["y"][1] - d["y"][0])
    def G(y, sigma_G, Pth):
        norm = 1 / (Ceq * sigma_G * sqrt(2 * np.pi))
        return norm * dy * np.exp(-((y - Pth) / (Ceq * sigma_G * sqrt(2)))**2)

    # Construct CF_L1, CF_L2, CF_L3, CF_R1, CF_R2, CF_R3 (Eq 121-4)
    # Cumulative probability functions around each sub-eye threshold
    for side in ("left", "right"):
        h = hist["F_"+side]
        for Pth, subeye in zip((Pth1, Pth2, Pth3), ("1", "2", "3")):
            cf = np.zeros(len(h))
            pth_idx = np.argmin(np.abs(Pth - d["y"]))
            cf[pth_idx : ] = np.cumsum(h[pth_idx :])  # Integrate upwards
            cf[: pth_idx] = np.flip(np.cumsum(np.flip(h[: pth_idx])))  # Integrate downwards
            hist["CF_"+side+"_"+subeye] = cf

        def get_ser(sigma_G, side=side):
            ser1 = np.dot(hist["CF_"+side+"_1"], G(d["y"], sigma_G, Pth1))
            ser2 = np.dot(hist["CF_"+side+"_2"], G(d["y"], sigma_G, Pth2))
            ser3 = np.dot(hist["CF_"+side+"_3"], G(d["y"], sigma_G, Pth3))
            return ser1 + ser2 + ser3

        sigma_guess = msmts["oma_outer"]/3/10
        sol = root(lambda sigma: (np.log10(get_ser(abs(sigma))) - np.log10(TARGET_SER)), x0=sigma_guess, tol=1e-6)
        sigmas.append(abs(sol.x[0]))

    N = min(sigmas) # We want the largest of the left/right SERs to be equal to the TARGET_SER
    return np.sqrt(N**2 + S**2)