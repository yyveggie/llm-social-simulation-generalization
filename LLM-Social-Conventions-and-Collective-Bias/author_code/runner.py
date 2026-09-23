import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import simulation_module as sm
import yaml
from munch import munchify
import utils as ut
import prompting as pr
from pathlib import Path

with open(os.environ.get("SOCIAL_CONFIG", "config.yaml"), "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

N = config.params.N
runs = config.params.runs
rewards_set = config.params.rewards_set
memory_size_set = config.params.memory_size_set
initial = config.params.initial
total_interactions = config.params.total_interactions
temperature = config.params.temperature
shorthand = config.model.shorthand
options_set = config.params.options_set
minority_size_set = config.minority.minority_size_set
version = config.sim.version
experiments = getattr(config, "experiments", munchify({}))
run_individual_bias = getattr(experiments, "individual_bias", False)
run_collective_convergence = getattr(experiments, "collective_convergence", False)
run_committed_minority = getattr(experiments, "committed_minority", False)
individual_repeats = getattr(experiments, "individual_repeats", 5000)


OUTDIR = os.environ.get("SOCIAL_OUTDIR", "data")
os.makedirs(OUTDIR, exist_ok=True)


def _parallel_runs():
    v = getattr(config.params, "parallel_runs", 4)
    try:
        p = max(1, int(v))
    except (TypeError, ValueError):
        p = 1
    try:
        ap = config.api.active_provider
        mc = int(getattr(config.api.providers[ap], "max_concurrency", 0) or 0)
    except Exception:
        mc = 0
    if mc:
        p = min(p, max(1, mc // 2))
    return max(1, min(p, runs))


def _run_parallel(fn, items):
    items = list(items)
    parallel = _parallel_runs()
    if parallel <= 1 or len(items) <= 1:
        for it in items:
            fn(it)
        return
    print(f"[并行] run 级并行度 = {parallel}（共 {len(items)} 个 run）", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for f in futures:
            try:
                f.result()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]


def _migrate_legacy_temp(mainfname, frame, options, required_interactions=None):
    base = os.path.basename(mainfname)
    legacy = os.path.join(OUTDIR, "temporary_" + base)
    if not os.path.exists(legacy):
        return
    for run in range(runs):
        done = run in frame and ut.is_completed_population_dataframe(
            frame[run], options, required_interactions=required_interactions)
        if not done:
            target = os.path.join(OUTDIR, f"temporary_run{run}_" + base)
            if not os.path.exists(target):
                try:
                    os.replace(legacy, target)
                    print(f"[并行] 迁移遗留 checkpoint {os.path.basename(legacy)} -> {os.path.basename(target)}", flush=True)
                except OSError:
                    pass
            return


def individual_bias_runner():
    for rewards in rewards_set:
        for m in memory_size_set:
            for options in options_set:

                mainfname = f"{OUTDIR}/{shorthand}_no_memory_bias_test_{''.join([str(m) for m in options])}_{m}mem" + ".pkl"
                print(mainfname)
                try:
                    mainframe = pickle.load(open(mainfname, 'rb'))
                except:
                    mainframe = {'simulation': ut.get_player(), 'tracker': {'answers': []}}
                ut.compact_individual_dataframe(mainframe, options)
                sm.individual(dataframe=mainframe, memory_size=m, rewards = rewards, repeats=individual_repeats, options = options, fname = mainfname)
                ut.save_pickle(mainframe, mainfname)

def collective_convergence_runner():
    for rewards in rewards_set:
        for m in memory_size_set:
            for options in options_set:
                mainfname = '.pkl'
                if initial == 'None':
                    mainfname = f"{OUTDIR}/{shorthand}_converged_baseline_{'_'.join(options)}_{rewards[0]}_{rewards[1]}_{m}mem_{config.network.network_type}_{N}ps_{temperature}tmp.pkl"

                else:

                    mainfname = f"{OUTDIR}/{shorthand}_evolved_from_{initial}_{'_'.join(options)}_{rewards[0]}_{rewards[1]}_{m}mem_{config.network.network_type}_{N}ps_{total_interactions}ints_{temperature}tmp.pkl"
                print(mainfname)
                mainframe = ut.load_mainframe(mainfname)
                mainframe['rules'] = pr.get_rules(rewards, options = options)
                lock = threading.Lock()
                required = None if initial == 'None' else total_interactions
                _migrate_legacy_temp(mainfname, mainframe, options, required_interactions=required)


                def _one_run(run, rewards=rewards, m=m, options=options, mainfname=mainfname):
                    temp_fname = os.path.join(OUTDIR, f"temporary_run{run}_" + os.path.basename(mainfname))
                    if initial == 'None':
                        with lock:
                            if run in mainframe and ut.is_completed_population_dataframe(mainframe[run], options):


                                sm.mark_population_run_done(run, runs)
                                return
                            mainframe.pop(run, None)
                        print("---------- BASELINE CONVERGENCE ----------")
                        df = ut.get_empty_population(fname=temp_fname, options=options)
                        sm.population(dataframe=df, run=run, memory_size=m, rewards=rewards, options=options, fname=temp_fname, runs=runs)
                    else:
                        with lock:
                            if run in mainframe and ut.is_completed_population_dataframe(mainframe[run], options, required_interactions=total_interactions):
                                sm.mark_committed_run_done(run, runs)
                                return
                            mainframe.pop(run, None)
                        df = ut.get_prepared_population(fname=temp_fname, rewards=rewards, options=options, minority_size=0, memory_size=m)
                        print("---------- CONTINUING EVOLUTION ----------")
                        print(f"--- STARTING RUN {run} ---")
                        sm.committed(dataframe=df, run=run, memory_size=m, rewards=rewards, options=options, fname=temp_fname, total_interactions=total_interactions, runs=runs)
                    print(run)

                    with lock:
                        mainframe[run] = df
                        ut.save_pickle(mainframe, mainfname)

                    Path(temp_fname).unlink(missing_ok=True)

                _run_parallel(_one_run, range(runs))

def committed_runner():
    for rewards in rewards_set:
        for memory_size in memory_size_set:
            for cm in minority_size_set:
                for options in options_set:
                    if initial == 'None':
                        raise ValueError(
                            "无法运行 committed_minority：该实验必须基于已经收敛的 baseline 群体。\n"
                            "请先在 config.yaml 中设置 experiments.collective_convergence: True、experiments.committed_minority: False、params.initial: 'None'，运行 python runner.py 生成 baseline .pkl。\n"
                            "baseline 生成后，再设置 experiments.collective_convergence: False、experiments.committed_minority: True，并把 params.initial 设置为要使用的 baseline run 编号，例如 0。"
                        )

                    cmfname = f"{OUTDIR}/{shorthand}_70b_{version}_{initial}_{cm}cmtd_{'_'.join(options)}_{rewards[0]}_{rewards[1]}_{memory_size}mem_{config.network.network_type}_{N}ps_{temperature}tmp.pkl"
                    print(cmfname)
                    cmframe = ut.load_mainframe(fname=cmfname)
                    print("cmframe keys:", cmframe.keys())
                    lock = threading.Lock()
                    _migrate_legacy_temp(cmfname, cmframe, options, required_interactions=total_interactions)


                    def _one_run(run, rewards=rewards, memory_size=memory_size, cm=cm, options=options, cmfname=cmfname):

                        with lock:
                            if run in cmframe and ut.is_completed_population_dataframe(cmframe[run], options, required_interactions=total_interactions):

                                sm.mark_committed_run_done(run, runs)
                                return
                            cmframe.pop(run, None)

                        mainframe = ut.get_prepared_population(fname='.pkl', rewards=rewards, options=options, minority_size=0, memory_size=memory_size)
                        temp_fname = os.path.join(OUTDIR, f"temporary_run{run}_" + os.path.basename(cmfname))

                        df = ut.load_mainframe(fname=temp_fname)

                        if len(df.keys()) == 0 or not ut.is_valid_population_dataframe(df, options):
                            print(f'----------STARTING RUN {run} FROM SCRATCH----------')
                            df = mainframe


                            if version == 'swap':
                                print("---------- SWAPPING COMMITTED AGENTS ----------")
                                df = ut.swap_committed(df, cm)

                            if version == 'inject':
                                print("---------- ADDING COMMITTED AGENTS ----------")
                                df = ut.add_committed(df, cm)

                        print(f"Run: {run}")
                        print(f"Initial population: {N}")
                        print(f"There are {len(df['simulation'].keys())} players in the game")
                        print(f"minority size: {cm}")
                        word =  df['convergence']['committed_to']
                        print(f'committment word is: {word}')
                        committed_agent_ids = [player for player in df['simulation'].keys() if df['simulation'][player]['committed_tag'] == True]
                        print(f"There are {len(committed_agent_ids)} committed agents: {committed_agent_ids}")

                        print("---------- RUNNING COMMITTED AGENTS ----------")
                        sm.committed(dataframe=df, run=run, memory_size=memory_size, rewards=rewards, options=options, fname=temp_fname, total_interactions=total_interactions, runs=runs)


                        with lock:
                            cmframe[run] = df
                            ut.save_pickle(cmframe, cmfname)


                        Path(temp_fname).unlink(missing_ok=True)

                    _run_parallel(_one_run, range(runs))


selected_experiments = []
if run_individual_bias:
    selected_experiments.append("individual_bias")
if run_collective_convergence:
    selected_experiments.append("collective_convergence")
if run_committed_minority:
    selected_experiments.append("committed_minority")
if len(selected_experiments) == 0:
    raise ValueError("No experiments enabled. Set at least one experiments.* flag to True in config.yaml.")
print("Enabled experiments:", selected_experiments)

if run_individual_bias:
    individual_bias_runner()
if run_collective_convergence:
    collective_convergence_runner()
if run_committed_minority:
    committed_runner()


