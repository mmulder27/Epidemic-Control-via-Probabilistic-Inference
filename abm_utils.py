import pandas as pd
import numpy as np
import covid19
from COVID19.model import Model, Parameters
import COVID19.simulation as simulation
from pathlib import Path
import logging


def status_to_state_(status):
    return ((status > 0) and (status < 6)) + 2 * (status >=6)

status_to_state = np.vectorize(status_to_state_)

def listofhouses(houses):
    housedict = {house_no : [] for house_no in np.unique(houses)}
    for i in range(len(houses)):
        housedict[houses[i]].append(i)
    return housedict

def free_abm(params,
             N,
             T,
             logger = None,
             patient_zeroes = 50,
             input_parameter_file = None,
             household_demographics_file = None,
             parameter_line_number = 1,
             name_file_res = "res",
             output_dir = None,
             save_every_iter = 5,
             stop_zero_I = True,
             data = {},
             callback = lambda data : None
            ):
    if logger is None:
        logger = logging.getLogger("dummy")
        logger.addHandler(logging.NullHandler())

    # Get base path relative to this script
    BASE = Path(__file__).resolve().parent

    # Use defaults if not provided
    if input_parameter_file is None:
        input_parameter_file = BASE / "OpenABM-Covid19" / "tests" / "data" / "baseline_parameters.csv"
    if household_demographics_file is None:
        household_demographics_file = BASE / "OpenABM-Covid19" / "tests" / "data" / "baseline_household_demographics.csv"
    if output_dir is None:
        output_dir = BASE / "results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize parameter model
    try:
        params_model = Parameters(
            str(input_parameter_file),
            parameter_line_number,
            str(output_dir),
            str(household_demographics_file)
        )
    except Exception as e:
        print("Failed to construct Parameters:", e)


    for k, val in params.items():
        params_model.set_param(k, val)
    params_model.set_param("end_time", T)
    params_model.set_param("n_total", N)
    params_model.set_param("n_seed_infection", patient_zeroes)
    print(params_model)


    model = simulation.COVID19IBM(model=Model(params_model))
    T = params_model.get_param("end_time")
    N = params_model.get_param("n_total")
    print(N)

    sim = simulation.Simulation(env=model, end_time=T, verbose=False)
    

    for col_name in ["I", "IR"]:
        data[col_name] = np.full(T, np.nan)
    

    for t in range(T):
        sim.steps(1)
        sim.collect_results = lambda state, action: None
        print(f"Step: {t}")
        status = np.array(covid19.get_state(model.model.c_model))
        state = status_to_state(status)
        nS, nI, nR = (state == 0).sum(), (state == 1).sum(), (state == 2).sum()

        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break

        logger.info(f'time: {t}')
        data["I"][t] = nI
        data["IR"][t] = nI + nR

    print("End of Simulation")
    # sim.results.clear()
    # sim.results_all_simulations.clear()
    # sim.env.model = None     # drop the COVID19IBM → Model
    # sim.env    = None 
    # del sim
    # del model
    # del params_model
    # gc.collect()
    return data


def loop_abm(params,
             N,
             T,
             inference_algorithm,
             logger=None,
             input_parameter_file=None,
             household_demographics_file=None,
             parameter_line_number=1,
             seed=1,
             initial_steps=10,
             num_test_random=0,
             num_test_algo=50,
             fraction_SM_obs=0.5,
             fraction_SS_obs=1.0,
             quarantine_HH=False,
             test_HH=False,
             name_file_res="res",
             output_dir=None,
             save_every_iter=5,
             stop_zero_I=True,
             adoption_fraction=1.0,
             fp_rate=0.0,
             fn_rate=0.0,
             smartphone_users_abm=False,
             callback=lambda x: None,
             data=None,
             patient_zeroes = 50):
    

    if logger is None:
        logger = logging.getLogger("dummy")
        logger.addHandler(logging.NullHandler())

    if data is None:
        data = {}

    BASE = Path(__file__).resolve().parent

    if input_parameter_file is None:
        input_parameter_file = BASE / "OpenABM-Covid19" / "tests" / "data" / "baseline_parameters.csv"
    if household_demographics_file is None:
        household_demographics_file = BASE / "OpenABM-Covid19" / "tests" / "data" / "baseline_household_demographics.csv"
    if output_dir is None:
        output_dir = BASE / "results"

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        params_model = Parameters(
            str(input_parameter_file),
            parameter_line_number,
            str(output_dir),
            str(household_demographics_file)
        )
    except Exception as e:
        logger.error("Failed to construct Parameters: %s", e)
        raise

    for k, val in params.items():
        params_model.set_param(k, val)

    params_model.set_param("end_time", T)
    params_model.set_param("n_total", N)
    params_model.set_param("n_seed_infection", patient_zeroes)

    rng = np.random.RandomState(seed)
    model = simulation.COVID19IBM(model=Model(params_model))
    sim = simulation.Simulation(env=model, end_time=T, verbose=False)

    houses = covid19.get_house(model.model.c_model)
    # housedict = {h: list(np.where(houses == h)[0]) for h in np.unique(houses)}
    def listofhouses(houses):
        housedict = {house_no : [] for house_no in np.unique(houses)}
        for i in range(len(houses)):
            housedict[houses[i]].append(i)
        return housedict
    housedict = listofhouses(houses)

    has_app = covid19.get_app_users(model.model.c_model) if smartphone_users_abm else np.ones(N, dtype=int)
    has_app &= (rng.random(N) <= adoption_fraction)

    inference_algorithm.init(N, T)
    print(N)
    indices = np.arange(N)
    excluded = np.zeros(N, dtype=bool)
    noise_SM = rng.random(N)
    noise_SS = rng.random(N) if fraction_SS_obs < 1 else None


    for col_name in ["I", "IR"]:
        data[col_name] = np.full(T, np.nan)

    #data["logger"] = logger

    num_quarantined = fp_num = fn_num = p_num = n_num = freebirds = 0
    daily_obs = []
    nfree = params_model.get_param("n_seed_infection")
    print("ran1")
    for t in range(T):
        sim.steps(1)
        sim.collect_results = lambda state, action: None
        print(f"Step: {t}")
        status = np.array(covid19.get_state(model.model.c_model))
        state = status_to_state(status)

        nS, nI, nR = (state == 0).sum(), (state == 1).sum(), (state == 2).sum()

        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break
        if t == initial_steps:
            logger.info("Inference algorithm starts now.")
        logger.info(f"time: {t}")
        import time

        daily_contacts = covid19.get_contacts_daily(model.model.c_model, t)
        t2 = time.time_ns()
        weighted_contacts = [(c[0], c[1], t, 2.0 if c[3] == 0 else 1.0) for c in daily_contacts if has_app[c[0]] and has_app[c[1]]]
        t3 = time.time_ns()
        #print(f"Contacts for day {t} retrieved in {(t2 - t1) / 1e6:.2f} ms, weighted contacts in {(t3 - t2) / 1e6:.2f} ms")


        if fp_rate or fn_rate:
            noise = rng.random(N)
            f_state = (state == 1) * (noise > fn_rate) + (state == 0) * (noise < fp_rate) + 2 * (state == 2)
        else:
            f_state = state

        to_quarantine = []
        all_test = []
        excluded_now = excluded.copy()
        fp_today = fn_today = p_today = n_today = 0

        def test_and_quarantine(rank, num):
            nonlocal to_quarantine, excluded_now, all_test, fp_today, fn_today, p_today, n_today
            selected = []
            for i in rank:
                if len(selected) >= num:
                    break
                if excluded_now[i]:
                    continue
                selected.append(i)
                if f_state[i] == 1:
                    p_today += 1
                    if state[i] != 1:
                        fp_today += 1
                    q = housedict[houses[i]] if quarantine_HH else [i]
                    excluded_now[q] = True
                    to_quarantine.extend(q)
                    excluded[q] = True
                    all_test.extend(q if test_HH else [i])
                else:
                    n_today += 1
                    if state[i] == 1:
                        fn_today += 1
                    excluded_now[i] = True
                    all_test.append(i)
            return selected

        if t < initial_steps:
            daily_obs = []
            num_test_algo_today = 0
        else:
            num_test_algo_today = num_test_algo


        if (nfree == 0 and quarantine_HH) or t < initial_steps:
            inference_algorithm.update_history(weighted_contacts, daily_obs, t)
            rank = np.random.permutation(N).tolist()
            continue
        else:
            rank = inference_algorithm.rank(t, weighted_contacts, daily_obs, data)

        test_algo = test_and_quarantine(rank, num_test_algo_today)

        SS = indices[(status == 4) & (noise_SS < fraction_SS_obs)] if fraction_SS_obs < 1 else indices[status == 4]
        SM = indices[(status == 5) & (noise_SM < fraction_SM_obs)]

        SS = test_and_quarantine(SS, len(SS))
        SM = test_and_quarantine(SM, len(SM))
        test_random = test_and_quarantine(rng.permutation(N), num_test_random)

        num_quarantined += len(to_quarantine)
        print(to_quarantine)
        #q_list = [int(x) for x in np.asarray(to_quarantine).ravel() if x >= 0]
        #covid19.intervention_quarantine_list(model.model.c_model, q_list, T+1)
        covid19.intervention_quarantine_list(model.model.c_model, to_quarantine, T + 1)

        daily_obs = [(i, f_state[i], t-1) for i in all_test]
        excluded[[i for i, result, _ in daily_obs if result == 2]] = True

        data["I"][t] = nI
        data["IR"][t] = nI + nR
    
        ninfq = sum(state[i] > 0 for i in to_quarantine)
        nfree = int(nI - sum(excluded[state == 1]))

        fp_num += fp_today
        fn_num += fn_today
        p_num += p_today
        n_num += n_today
        freebirds = nfree


    print("End of Intervention Simulation")
    # sim.results.clear()
    # sim.results_all_simulations.clear()
    # del sim
    # del model
    # del params_model
    # gc.collect()
    return data

