import batman
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit


def fit_light_curve(data, transit_params, systematic_params, check_guess=False, draw_fits=False, print_results=False):
    # Unpack transit parameters.
    p0_guess = []
    free_param_names = []
    fixed_params_names = []
    for p_key, p_val in transit_params.items():
        if p_key == "ld_law":
            ld_law_label = p_val
        else:
            if not p_val["fixed"]:
                p0_guess.append(p_val["value"])
                free_param_names.append(p_key)
            else:
                fixed_params_names.append(p_key)

    # Unpack systematic parameters.
    for p_key, p_val in systematic_params.items():
        if p_key == "systematic_model_label":
            systematic_model_label = p_val
        else:
            if not p_val["fixed"]:
                p0_guess.append(p_val["value"])
                free_param_names.append(p_key)
            else:
                fixed_params_names.append(p_key)

    # Iterate fitting with updating mask for outliers.
    _fit_iter = 0
    no_mask = np.ones(data["times"].shape).astype(bool)
    time_mask = np.ones(data["times"].shape).astype(bool)
    while True:

        def model(_, *theta, use_mask=True):
            m_mask = time_mask if use_mask else no_mask
            lc = transit_model(data, theta, free_param_names, transit_params, ld_law_label, m_mask)
            sys = systematic_model(data, theta, free_param_names, systematic_params, systematic_model_label, m_mask)
            return lc * sys

        if _fit_iter == 0 and check_guess:
            # Check guess.
            guessed_model = model(None, *p0_guess, use_mask=False)
            sys_model = systematic_model(data, p0_guess, free_param_names, systematic_params, systematic_model_label, no_mask)
            _draw_model(data, guessed_model, sys_model, time_mask, "Initial Guess")

        # Fit.
        popt, pcov = curve_fit(
            model, data["times"][time_mask], data["flux"][time_mask],
            sigma=data["flux_err"][time_mask], p0=p0_guess, method='lm')
        _fit_iter += 1

        # Determine if outliers.
        fitted_model = model(None, *popt, use_mask=False)
        residuals = data["flux"] - fitted_model
        deviation = np.abs(residuals) / np.std(residuals)
        deviation = np.ma.array(deviation, mask=~time_mask)
        max_deviation_idx = np.ma.argmax(deviation)
        if deviation[max_deviation_idx] > data["outlier_threshold"]:
            # Update outliers mask.
            time_mask[max_deviation_idx] = False
            continue
        else:
            # No outliers, end fitting.
            break

    # Pack results.
    results = {}
    free_params = {}
    perr = np.sqrt(np.diag(pcov))
    best_fit_model = model(None, *popt, use_mask=False)
    best_sys_model = systematic_model(data, popt, free_param_names, systematic_params, systematic_model_label, no_mask)
    for fp_idx, fp_name in enumerate(free_param_names):
        free_params[fp_name] = popt[fp_idx]
        free_params["{}_err".format(fp_name)] = perr[fp_idx]
        if print_results == True:
            print("{} = {} +/- {}".format(fp_name, popt[fp_idx], perr[fp_idx]))
    results["light_curve_model"] = best_fit_model
    results["systematic_model"] = best_sys_model
    results["mask"] = time_mask
    results["residual"] = residuals
    results["corrected_flux"] = data["flux"] - best_sys_model + 1
    results["corrected_flux_error"] = data["flux_err"]
    results["free_param_values"] = free_params
    results["free_param_names"] = free_param_names


    #print('Rp/Rs={} +- {}'.format(rp, rp_err))
    #print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))
    #print('t0={}'.format(popt[0]))
    #print('a={}'.format(popt[2]))
    #print('inc={}'.format(popt[3]))
    print('Residuals={} ppm'.format(np.std(residuals) * 1.e6))
    print('Done in {} iterations.'.format(_fit_iter))


    if draw_fits:
        # Draw final fit.
        _draw_model(data, best_fit_model, best_sys_model, time_mask, "Light Curve Fit")

    return results


def get_param(label, theta, free_param_names, params):
    if label in free_param_names:
        return theta[free_param_names.index(label)]
    else:
        return params[label]["value"]


def transit_model(data, theta, free_param_names, transit_params, ld_law_label, mask):
    batman_params = batman.TransitParams()
    batman_params.t0 = get_param("t0", theta, free_param_names, transit_params)
    batman_params.per = get_param("period", theta, free_param_names, transit_params)
    batman_params.rp = get_param("rp", theta, free_param_names, transit_params)
    batman_params.a = get_param("a", theta, free_param_names, transit_params)
    batman_params.inc = get_param("inc", theta, free_param_names, transit_params)
    batman_params.ecc = get_param("ecc", theta, free_param_names, transit_params)
    batman_params.w = get_param("omega", theta, free_param_names, transit_params)

    if ld_law_label == "quadratic":
        batman_params.u = [get_param("u1", theta, free_param_names, transit_params),
                           get_param("u2", theta, free_param_names, transit_params)]
        batman_params.limb_dark = "quadratic"
    elif ld_law_label == "kipping":
        q1 = get_param("u1", theta, free_param_names, transit_params)
        q2 = get_param("u2", theta, free_param_names, transit_params)
        batman_params.u = [2. * q1**0.5 * q2, q1**0.5 * (1. - 2. * q2)]
        batman_params.limb_dark = "quadratic"
    elif ld_law_label == "squareroot":
        batman_params.u = [get_param("u1", theta, free_param_names, transit_params),
                           get_param("u2", theta, free_param_names, transit_params)]
        batman_params.limb_dark = "squareroot"
    elif ld_law_label == "linear":
        batman_params.u = [get_param("u1", theta, free_param_names, transit_params)]
        batman_params.limb_dark = "linear"
    elif ld_law_label == "nonlinear":
        batman_params.u = [get_param("u1", theta, free_param_names, transit_params),
                           get_param("u2", theta, free_param_names, transit_params),
                           get_param("u3", theta, free_param_names, transit_params),
                           get_param("u4", theta, free_param_names, transit_params)]
        batman_params.limb_dark = "nonlinear"
    else:
        raise ValueError("ld_law_label not recognised.")

    m = batman.TransitModel(batman_params, data["times"])
    return m.light_curve(batman_params)[mask]


def systematic_model(data, theta, free_param_names, systematic_params, systematic_model_label, mask):
    if systematic_model_label == "xy_mirror_tilt":
        return _sys_model_xy_mirror_tilt(data, theta, free_param_names, systematic_params)[mask]
    elif systematic_model_label == "x_y":
        return _sys_model_x_y(data, theta, free_param_names, systematic_params)[mask]
    elif systematic_model_label == "xy":
        return _sys_model_xy(data, theta, free_param_names, systematic_params)[mask]
    elif systematic_model_label == "xyt2":
        return _sys_model_xyt2(data, theta, free_param_names, systematic_params)[mask]
    elif systematic_model_label == "ramp":
        return _sys_model_ramp(data, theta, free_param_names, systematic_params)[mask]
    else:
        raise ValueError("systematic_model_label not recognised.")


def _sys_model_x_y(data, theta, free_param_names, systematic_params):
    s0 = get_param("s0", theta, free_param_names, systematic_params)
    s1 = get_param("s1", theta, free_param_names, systematic_params)
    s2 = get_param("s2", theta, free_param_names, systematic_params)
    s3 = get_param("s3", theta, free_param_names, systematic_params)
    sys = s0 + (s1 * data["x"]) + (s2 * data["y"]) + (s3 * data["times"])
    return sys


def _sys_model_xy(data, theta, free_param_names, systematic_params):
    s0 = get_param("s0", theta, free_param_names, systematic_params)
    s1 = get_param("s1", theta, free_param_names, systematic_params)
    s2 = get_param("s2", theta, free_param_names, systematic_params)
    aby = abs(data["y"] - np.median(data["y"]))
    sys = s0 + (s1 * (data["x"] * aby)) + (s2 * data["times"])
    return sys


def _sys_model_xyt2(data, theta, free_param_names, systematic_params):
    s0 = get_param("s0", theta, free_param_names, systematic_params)
    s1 = get_param("s1", theta, free_param_names, systematic_params)
    s2 = get_param("s2", theta, free_param_names, systematic_params)
    s3 = get_param("s3", theta, free_param_names, systematic_params)
    aby = abs(data["y"] - np.median(data["y"]))
    sys = s0 + (s1 * (data["x"] * aby)) + (s2 * data["times"] + s3 * data["times"] * data["times"])
    return sys


def _sys_model_xy_mirror_tilt(data, theta, free_param_names, systematic_params):
    s1 = get_param("s1", theta, free_param_names, systematic_params)
    s2 = get_param("s2", theta, free_param_names, systematic_params)
    t_idx = get_param("t_idx", theta, free_param_names, systematic_params)
    t0 = get_param("t0", theta, free_param_names, systematic_params)
    t1 = get_param("t1", theta, free_param_names, systematic_params)
    sys = 1. + s1 * data["x"] + s2 * data["y"]
    sys[:t_idx] *= t0
    sys[t_idx:] *= t1
    return sys


def _sys_model_ramp(data, theta, free_param_names, systematic_params):
    s0 = get_param("s0", theta, free_param_names, systematic_params)
    s1 = get_param("s1", theta, free_param_names, systematic_params)
    s2 = get_param("s2", theta, free_param_names, systematic_params)
    s3 = get_param("s3", theta, free_param_names, systematic_params)
    sys = s0 - np.exp(-s1 * data["times"] + s2) + (s3 * data["times"])
    return sys


def _draw_model(data, total_model, sys_model, mask, title):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)

    ax1.set_title(title)
    ax1.errorbar(data["times"][mask], data["flux"][mask], yerr=data["flux_err"][mask], fmt='.', zorder=0, alpha=0.4)
    ax1.plot(data["times"], total_model)
    ax1.set_ylabel("Normalised Flux")

    ax2.errorbar(data["times"][mask], (data["flux"][mask] - sys_model[mask] + 1), yerr=data["flux_err"][mask], fmt='.', zorder=0, alpha=0.4)
    ax2.plot(data["times"][mask], total_model[mask] - sys_model[mask] + 1)
    ax2.set_ylabel("Flux - Systematic Model")

    ax3.set_xlabel("Time (BJD)")
    ax3.axhline(0, ls=':', color='k')
    ax3.set_ylabel("Residuals (ppm)")
    ax3.errorbar(data["times"][mask], (data["flux"][mask] - total_model[mask]) * 1e6, yerr=(data["flux_err"][mask]) * 1e6, fmt='.')

    plt.tight_layout()
    plt.show()


    fig = plt.figure(figsize=(9, 3))
    plots = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[4, 1])
    ax1 = fig.add_subplot(plots[0])
    ax2 = fig.add_subplot(plots[1])
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax1.set_ylabel("Residuals (ppm)")
    ax1.set_xlabel("Time (BJD)")
    ax1.errorbar(data['times'][mask], (data['flux'][mask] - total_model[mask])*1e6,
                 yerr=(data['flux_err'][mask])*1e6, fmt='.')
    ax1.axhline(0, ls=':', color='k')
    ax2.hist((data['flux'][mask]-total_model[mask])*1e6, orientation='horizontal', bins=15)
    ax2.axhline(0, ls=':', color='k')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


def get_limb_dark(sld, wvl_min, wvl_max, transit_params,
                  instrument, law, u1_fixed, u2_fixed, u3_fixed, u4_fixed):

#     sld = StellarLimbDarkening(
#         M_H=M_H, Teff=Teff, logg=logg, ld_model=ld_model,
#         ld_data_path=ld_data_path)


    #Theoretical limb-darkening parameters.
    if law == 'nonlinear':
        u1, u2, u3, u4 = sld.compute_4_parameter_non_linear_ld_coeffs(
            wavelength_range=(wvl_min,wvl_max),
            mode=instrument)
            #mode='custom',
            #custom_wavelengths=ds_['wavelength'].values * 1.e4,
            #custom_throughput=custom_throughput)
        transit_params["ld_law"] = law
        if u1_fixed == True:
            transit_params["u1"] = {"value": u1, "fixed": True}
        else:
            transit_params["u1"] = {"value": u1_fixed,  "fixed": False}
        if u2_fixed == True:
            transit_params["u2"] = {"value": u2, "fixed": True}
        else:
            transit_params["u2"] = {"value": u2_fixed,  "fixed": False}
        if u3_fixed == True:
            transit_params["u3"] = {"value": u3, "fixed": True}
        else:
            transit_params["u3"] = {"value": u3_fixed,  "fixed": False}
        if u4_fixed == True:
            transit_params["u4"] = {"value": u4, "fixed": True}
        else:
            transit_params["u4"] = {"value": u4_fixed,  "fixed": False}
    if law == 'linear':
        u1 = sld.compute_linear_ld_coeffs(
            wavelength_range=(wvl_min,wvl_max),
            mode=instrument)
            #mode='custom',
            #custom_wavelengths=ds_['wavelength'].values * 1.e4,
            #custom_throughput=custom_throughput)
        transit_params["ld_law"] = law
        if u1_fixed == True:
            transit_params["u1"] = {"value": u1, "fixed": True}
        else:
            transit_params["u1"] = {"value": u1_fixed,  "fixed": False}
    if law == "quadratic": 
        u1, u2 = sld.compute_quadratic_ld_coeffs(
            wavelength_range=(wvl_min,wvl_max),
            mode=instrument)
            #mode='custom',
            #custom_wavelengths=ds_['wavelength'].values * 1.e4,
            #custom_throughput=custom_throughput)
        transit_params["ld_law"] = law
        if u1_fixed == True:
            transit_params["u1"] = {"value": u1, "fixed": True}
        else:
            transit_params["u1"] = {"value": u1_fixed,  "fixed": False}
        if u2_fixed == True:
            transit_params["u2"] = {"value": u2, "fixed": True}
        else:
            transit_params["u2"] = {"value": u2_fixed,  "fixed": False}
    if law == "squareroot":
        u1, u2 = sld.compute_squareroot_ld_coeffs(
            wavelength_range=(wvl_min,wvl_max),
            mode=instrument)
            #mode='custom',
            #custom_wavelengths=ds_['wavelength'].values * 1.e4,
            #custom_throughput=custom_throughput)
        transit_params["ld_law"] = law
        if u1_fixed == True:
            transit_params["u1"] = {"value": u1, "fixed": True}
        else:
            transit_params["u1"] = {"value": u1_fixed,  "fixed": False}
        if u2_fixed == True:
            transit_params["u2"] = {"value": u2, "fixed": True}
        else:
            transit_params["u2"] = {"value": u2_fixed,  "fixed": False}
    if law == "kipping":
        u1, u2 = sld.compute_kipping_ld_coeffs(
            wavelength_range=(wvl_min,wvl_max),
            mode=instrument)
            #mode='custom',
            #custom_wavelengths=ds_['wavelength'].values * 1.e4,
            #custom_throughput=custom_throughput)
        transit_params["ld_law"] = law
        if u1_fixed == True:
            transit_params["u1"] = {"value": u1, "fixed": True}
        else:
            transit_params["u1"] = {"value": u1_fixed,  "fixed": False}
        if u2_fixed == True:
            transit_params["u2"] = {"value": u2, "fixed": True}
        else:
            transit_params["u2"] = {"value": u2_fixed,  "fixed": False}
            
            
    return(transit_params)


def check_systematic_params(systematic_params):
    
    param_list=[]

    if systematic_params['systematic_model_label']=="xy":
        for key, value in systematic_params.items():
            if key not in ["systematic_model_label","s0", "s1", "s2"]:
                raise ValueError("incorrect systematic model variables. model label 'xy' supports s0, s1, s2")
            param_list.append(key)
        if param_list != ["systematic_model_label", "s0", "s1", "s2"]:
            raise ValueError("a key is missing from systematic_params. model label 'xy' requires s1, s2")
    
    if systematic_params['systematic_model_label']=="x_y":
        for key, value in systematic_params.items():
            if key not in ["systematic_model_label","s0", "s1", "s2", "s3"]:
                raise ValueError("incorrect systematic model variables. model label x_y requires s0, s1, s2, s3")
            param_list.append(key)
        if param_list != ["systematic_model_label", "s0", "s1", "s2", "s3"]:
            raise ValueError("a key is missing from systematic_params. model label 'x_y' requires s0, s1, s2, s3")
    
    if systematic_params['systematic_model_label']=="xyt2":
        for key, value in systematic_params.items():
            if key not in ["systematic_model_label","s0", "s1", "s2", "s3"]:
                raise ValueError("incorrect systematic model variables. model label 'xyt2' requires s0, s1, s2, s3")
            param_list.append(key)
        if param_list != ["systematic_model_label", "s0", "s1", "s2", "s3"]:
            raise ValueError("a key is missing from systematic_params. model label 'xyt2' requires s0, s1, s2, s3")
    
    if systematic_params['systematic_model_label']=="xy_mirror_tilt":
        for key, value in systematic_params.items():
            if key not in ["systematic_model_label", "s1", "s2", "t_idx", "t0", "t1"]:
                raise ValueError("incorrect systematic model variables. model label 'xy_mirror_tilt' requires s1, s2, t_idx, t0, t1")
            param_list.append(key)
        if param_list != ["systematic_model_label", "s1", "s2", "t_idx", "t0", "t1"]:
            raise ValueError("a key is missing from systematic_params. model label 'xy_mirror_tilt' requires s1, s2, t_idx, t0, t1")
    
    if systematic_params['systematic_model_label']=="ramp":
        for key, value in systematic_params.items():
            if key not in ["systematic_model_label","s0", "s1", "s2", "s3"]:
                raise ValueError("incorrect systematic model variables. model label 'ramp' requires s0, s1, s2, s3")
            param_list.append(key)
        if param_list != ["systematic_model_label", "s0", "s1", "s2", "s3"]:
            raise ValueError("a key is missing from systematic_params. model label 'ramp' requires s0, s1, s2, s3")
    
    if systematic_params['systematic_model_label'] not in ["xy","x_y", "xyt2", "xy_mirror_tilt", "ramp"]:
        raise ValueError("systematic model not recognised. supported models are: xy, x_y, xyt2, xy_mirror_tilt, ramp")

        
def check_transit_params(transit_params):
    param_list = []
    for key, value in transit_params.items():
        if key not in ["t0","period", "rp", "a", "inc", "ecc", "omega"]:
            raise ValueError("incorrect transit parameters. light curve fitting requires t0, period, rp, a, inc, ecc, omega")
        param_list.append(key)
    if param_list != ["t0", "period", "rp", "a", "inc", "ecc", "omega"]:
        raise ValueError("a transit parameter is missing. light curve fitting requires t0, period, rp, a, inc, ecc, omega")