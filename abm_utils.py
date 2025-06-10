import pandas as pd
import numpy as np
import covid19
from COVID19.model import Model, Parameters
import COVID19.simulation as simulation
from pathlib import Path
import logging
import pickle


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
        callback(data)

    # Save outputs
    # sim.env.model.write_individual_file()
    # individual_file = output_dir / "individual_file_Run1.csv"
    # df_indiv = pd.read_csv(individual_file, skipinitialspace=True)
    # df_indiv.to_csv(output_dir / f"{name_file_res}_individuals.gz")

    # sim.env.model.write_transmissions()
    # transmission_file = output_dir / "transmission_Run1.csv"
    # df_trans = pd.read_csv(transmission_file)
    # df_trans.to_csv(output_dir / f"{name_file_res}_transmissions.gz")

    print("End of Simulation")
    sim.results.clear()
    sim.results_all_simulations.clear()
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

    data_states = {
        "true_conf": np.zeros((T, N)),
        "statuses": np.zeros((T, N)),
        "tested_algo": [],
        "tested_random": [],
        "tested_SS": [],
        "tested_SM": []
    }

    metrics = ["num_quarantined", "q_SS", "q_SM", "q_algo", "q_random", "infected_free", 
               "S", "I", "R", "IR", "aurI", "prec1%", "prec5%", 
               "test_+", "test_-", "test_f+", "test_f-"]
    for key in metrics:
        data[key] = np.full(T, np.nan)

    data["logger"] = logger

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
        data_states["true_conf"][t] = state
        data_states["statuses"][t] = status

        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break
        if t == initial_steps:
            logger.info("Inference algorithm starts now.")
        logger.info(f"time: {t}")
        import time

        t1 = time.time_ns()
        daily_contacts = covid19.get_contacts_daily(model.model.c_model, t)
        t2 = time.time_ns()
        weighted_contacts = [(c[0], c[1], c[2], 2.0 if c[3] == 0 else 1.0) for c in daily_contacts if has_app[c[0]] and has_app[c[1]]]
        t3 = time.time_ns()
        print(f"Contacts for day {t} retrieved in {(t2 - t1) / 1e6:.2f} ms, weighted contacts in {(t3 - t2) / 1e6:.2f} ms")


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
            #print(len(rank))
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

        inference_algorithm.update_history(weighted_contacts, daily_obs, t)

        if (nfree == 0 and quarantine_HH) or t < initial_steps:
            rank = np.random.permutation(N).tolist()
        else:
            rank = inference_algorithm.rank(t, data)

        test_algo = test_and_quarantine(rank, num_test_algo_today)


        #status_t = np.asarray(data_states["true_conf"][t])
        #indices2 = np.where(~excluded)[0]              # eligible individuals
        #labels = (status_t[indices2] == 1).astype(int) # 1 if infected, else 0
        #eventsI = list(zip(indices2, [t] * len(indices), labels))

        # _, yI, aurI, _ = roc_curve(dict(rank_algo), eventsI, lambda x: x)

        SS = indices[(status == 4) & (noise_SS < fraction_SS_obs)] if fraction_SS_obs < 1 else indices[status == 4]
        SM = indices[(status == 5) & (noise_SM < fraction_SM_obs)]

        SS = test_and_quarantine(SS, len(SS))
        SM = test_and_quarantine(SM, len(SM))
        test_random = test_and_quarantine(rng.permutation(N), num_test_random)

        num_quarantined += len(to_quarantine)
        covid19.intervention_quarantine_list(model.model.c_model, to_quarantine, T + 1)

        daily_obs = [(i, f_state[i], t) for i in all_test]
        excluded[[i for i, result, _ in daily_obs if result == 2]] = True

        data_states["tested_algo"].append(test_algo)
        data_states["tested_random"].append(test_random)
        data_states["tested_SS"].append(SS)
        data_states["tested_SM"].append(SM)

        data["S"][t], data["I"][t], data["R"][t] = nS, nI, nR
        data["IR"][t] = nI + nR
        #data["aurI"][t] = aurI
        #prec = lambda f: yI[int(f/100*len(yI))]/int(f/100*len(yI)) if len(yI) else np.nan
        ninfq = sum(state[i] > 0 for i in to_quarantine)
        nfree = int(nI - sum(excluded[state == 1]))
        #data["prec1%"][t] = prec(1)
        #data["prec5%"][t] = prec(5)
        data["num_quarantined"][t] = num_quarantined
        data["test_+"][t] = p_num
        data["test_-"][t] = n_num
        data["test_f+"][t] = fp_num
        data["test_f-"][t] = fn_num
        data["q_SS"][t] = len(SS)
        data["q_SM"][t] = len(SM)
        data["q_algo"][t] = sum(state[i] == 1 for i in test_algo)
        data["q_random"][t] = sum(state[i] == 1 for i in test_random)
        data["infected_free"][t] = nfree

        logger.info(f"True  : (S,I,R): ({nS:.1f}, {nI:.1f}, {nR:.1f})")
        #logger.info(f"AUR_I : {aurI:.3f}, prec(1%): {prec(1):.2f}, prec(5%): {prec(5):.2f}")
        logger.info(f"Tested algo (I): {data['q_algo'][t]}, random (I): {data['q_random'][t]}")
        logger.info(f"Free infected: {nfree}, Quarantined: {len(to_quarantine)}, Found: {ninfq}")

        fp_num += fp_today
        fn_num += fn_today
        p_num += p_today
        n_num += n_today
        freebirds = nfree

        callback(data)

    #     if t % save_every_iter == 0:
    #         pd.DataFrame.from_dict(data).to_csv(output_dir / f"{name_file_res}_res.gz")

    # pd.DataFrame.from_dict(data).to_csv(output_dir / f"{name_file_res}_res.gz")
    # with open(output_dir / f"{name_file_res}_states.pkl", "wb") as f_states:
    #     pickle.dump(data_states, f_states)

    # sim.env.model.write_individual_file()
    # df_indiv = pd.read_csv(output_dir / "individual_file_Run1.csv", skipinitialspace=True)
    # df_indiv.to_csv(output_dir / f"{name_file_res}_individuals.gz")

    # sim.env.model.write_transmissions()
    # df_trans = pd.read_csv(output_dir / "transmission_Run1.csv")
    # df_trans.to_csv(output_dir / f"{name_file_res}_transmissions.gz")

    print("End of Intervention Simulation")
    sim.results.clear()
    sim.results_all_simulations.clear()
    return data

