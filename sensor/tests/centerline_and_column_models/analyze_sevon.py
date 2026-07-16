import numpy as np
from scipy.optimize import curve_fit, least_squares

def _pseudo_voigt_profile(x, A, x0, sigma, gamma, p, B):
    """
    Pseudo-Voigt: A * [ p * L(x; x0,gamma) + (1-p) * G(x; x0,sigma) ] + B
    L and G are normalized to 1 at x=x0.
    """
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - x0) / gamma) ** 2)
    return A * (p * lorentz + (1.0 - p) * gauss) + B

def pseudo_voigt_fit(x, y,
                     p0=None,
                     bounds=None,
                     method='lsq',   # 'lsq' or 'robust'
                     robust_loss='huber', # used only if method=='robust'
                     robust_f_scale=1.0):
    """
    Fit a pseudo-Voigt profile to 1D data.

    Parameters
    - x: 1D array of independent variable (row indices or positions)
    - y: 1D array of intensities (same length as x)
    - p0: optional initial guess [A, x0, sigma, gamma, p, B]
          if None, sensible defaults are computed
    - bounds: optional tuple (lower_bounds, upper_bounds) for parameters
              default: A>0, sigma>0.5, gamma>0.5, p in [0,1]
    - method: 'lsq' uses scipy.curve_fit (non-robust),
              'robust' uses least_squares with loss (Huber/Tukey)
    - robust_loss: loss string passed to least_squares ('huber','soft_l1','cauchy','linear')
    - robust_f_scale: f_scale parameter for least_squares (controls outlier tolerance)

    Returns
    - result: dict with keys:
        'params': fitted [A, x0, sigma, gamma, p, B]
        'param_cov': covariance matrix (None if robust method used)
        'fwhm': estimated FWHM (approximate mix)
        'model': callable model(x) returning fitted curve
        'residuals': y - model(x)
        'success': boolean
        'message': optimizer message
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")

    # sensible defaults
    y_min, y_max = np.min(y), np.max(y)
    A0 = y_max - y_min
    x0_0 = x[np.argmax(y)]
    sigma0 = max(1.0, np.sum((x - x0_0)**2 * (y - y_min)) / (np.sum(y - y_min) + 1e-12))**0.5
    sigma0 = max(1.0, sigma0)
    gamma0 = max(1.0, sigma0 / 2.0)
    p0_default = [A0, x0_0, sigma0, gamma0, 0.5, y_min]

    if p0 is None:
        p0 = p0_default
    
    # default bounds
    if bounds is None:
        lower = [0.0, x.min(), 0.1, 0.1, 0.0, y_min - abs(A0)]
        upper = [np.inf, x.max(), (x.max()-x.min())*2, (x.max()-x.min())*2, 1.0, y_max + abs(A0)]
        bounds = (lower, upper)
    if p0 is not None:
        lb = np.asarray(bounds[0], dtype=float)
        ub = np.asarray(bounds[1], dtype=float)
        p0 = np.clip(p0, lb, ub)
    if method == 'lsq':
        try:
            popt, pcov = curve_fit(_pseudo_voigt_profile, x, y, p0=p0, bounds=bounds, maxfev=20000)
            success = True
            message = "curve_fit converged"
        except Exception as e:
            popt = np.array(p0)
            pcov = None
            success = False
            message = f"curve_fit failed: {e}"
    elif method == 'robust':
        # residual function for least_squares
        def resid(params):
            return _pseudo_voigt_profile(x, *params) - y

        # enforce bounds by parameter transform via least_squares bounds
        
        lb, ub = np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)
        # print(lb, ub, p0)
        res = least_squares(resid, x0=p0, bounds=(lb, ub), loss=robust_loss, f_scale=robust_f_scale, max_nfev=20000)
        popt = res.x
        pcov = None
        success = res.success
        message = res.message
    else:
        raise ValueError("method must be 'lsq' or 'robust'")

    # compute approximate FWHM for pseudo-Voigt: weighted sum of components' FWHM
    A_fit, x0_fit, sigma_fit, gamma_fit, p_fit, B_fit = popt
    fwhm_gauss = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_fit   # 2*sqrt(2 ln2) * sigma
    fwhm_lor = 2.0 * gamma_fit
    fwhm_mix = p_fit * fwhm_lor + (1.0 - p_fit) * fwhm_gauss

    model = lambda xx: _pseudo_voigt_profile(np.asarray(xx, dtype=float), *popt)
    residuals = y - model(x)

    return {
        'params': popt,
        'param_cov': pcov,
        'fwhm': float(fwhm_mix),
        'model': model,
        'residuals': residuals,
        'success': bool(success),
        'message': message
    }
def apply_psuedo_voigt_fit_across_columns(img):
    n = img.shape[1]
    xs = np.arange(n)
    reses = []
    success_mask = np.zeros(n, dtype = bool)
    params = np.zeros((n, 6), dtype=float)
    models = []
    cxs = np.arange(img.shape[0])
    for c in xs:
        col = img[:, c]
        if c > 0:
            p0 = params[c-1]
        else:
            p0 = None
        res = pseudo_voigt_fit(cxs, col, method='robust', robust_loss='huber', robust_f_scale=3.0, p0=p0)
        reses.append(res)
        success_mask[c] = res['success']
        params[c] = res['params']
        models.append(res['model'])
    return reses, success_mask, params, models

# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # synthetic test
    x = np.arange(0, 101)
    true = _pseudo_voigt_profile(x, A=200, x0=50, sigma=3.0, gamma=1.5, p=0.3, B=10)
    y = true + np.random.normal(scale=2.0, size=x.size)
    # add an absorption spike far away
    y[10:15] -= 30

    res = pseudo_voigt_fit(x, y, method='robust', robust_loss='huber', robust_f_scale=3.0)
    print("params:", res['params'])
    print("FWHM:", res['fwhm'], "success:", res['success'], res['message'])

    plt.plot(x, y, 'k.', label='data')
    plt.plot(x, res['model'](x), 'r-', label='pseudo-Voigt fit')
    plt.plot(x, true, 'b--', label='true (synthetic)')
    plt.legend()
    plt.show()
