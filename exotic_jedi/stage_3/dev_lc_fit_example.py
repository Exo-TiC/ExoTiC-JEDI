import batman
import numpy as np

from exotic_jedi.stage_3.dev_leastsqs_light_curve_fits import fit_light_curve


# Mock data.
batman_params = batman.TransitParams()
batman_params.t0 = 0.
batman_params.per = 3.
batman_params.rp = 0.1
batman_params.a = 7.
batman_params.inc = 88.
batman_params.ecc = 0.
batman_params.w = 90.
batman_params.u = [0.1, 0.3]
batman_params.limb_dark = "quadratic"
times = np.linspace(-0.15, 0.15, 500)
m = batman.TransitModel(batman_params, times)
sigma = 100.e-6
flux = m.light_curve(batman_params) + np.random.normal(loc=0., scale=sigma, size=len(times))
flux[100] = 1.02  # Make an outlier.
flux[300] = 0.98  # Make an outlier.
flux_err = sigma * np.ones_like(times)
x = np.ones_like(times)
y = np.ones_like(times)

# Fitting config.
data = {
    "times": times,
    "flux": flux,
    "flux_err": flux_err,
    "x": x,
    "y": y,
    "outlier_threshold": 5,
}
transit_params = {
    "t0": {"value": 0.0, "fixed": True},
    "period": {"value": 3.0, "fixed": True},
    "rp": {"value": 0.1, "fixed": False},
    "a": {"value": 7.1, "fixed": False},
    "inc": {"value": 88.0, "fixed": True},
    "ecc": {"value": 0.0, "fixed": True},
    "omega": {"value": 90.0, "fixed": True},
    "ld_law": "quadratic",
    "u1": {"value": 0.1, "fixed": True},
    "u2": {"value": 0.3, "fixed": True},
}
systematic_params = {
    "systematic_model_label": "x_y",
    "s0": {"value": 1.0, "fixed": False},
    "s1": {"value": 0.0, "fixed": True},
    "s2": {"value": 0.0, "fixed": True},
    "s3": {"value": 0.0, "fixed": True},
}

# Run fitting.
results = fit_light_curve(data, transit_params, systematic_params, check_guess=False, draw_fits=True)
