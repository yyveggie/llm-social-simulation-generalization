import random
import threading
import prompting as pr
import pickle
import yaml
from munch import munchify
import utils as ut
import meta_prompting as mp
from concurrent.futures import ThreadPoolExecutor

import os
with open(os.environ.get("SOCIAL_CONFIG", "config.yaml"), "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

total_interactions = config.params.total_interactions
N = config.params.N


def _emit_progress(current: int, total: int) -> None:
    if total > 0:
        print(f"[{current}/{total}]", flush=True)


POPULATION_PROGRESS_CAP = N * 30

_PROGRESS_LOCK = threading.Lock()
_PROGRESS_STATE = {"key": None, "runs": {}}


def _report_run_progress(run, done, per_run_total, total_runs):
    key = (per_run_total, total_runs)
    with _PROGRESS_LOCK:
        if _PROGRESS_STATE["key"] != key:

            _PROGRESS_STATE["key"] = key
            _PROGRESS_STATE["runs"] = {}
        entries = _PROGRESS_STATE["runs"]
        entries[run] = max(entries.get(run, 0), min(done, per_run_total))
        cur = sum(entries.values())
    _emit_progress(min(cur, per_run_total * total_runs), per_run_total * total_runs)


def mark_population_run_done(run, runs):
    _report_run_progress(run, POPULATION_PROGRESS_CAP, POPULATION_PROGRESS_CAP, max(1, runs))


def mark_committed_run_done(run, runs):
    _report_run_progress(run, total_interactions, total_interactions, max(1, runs))

if config.sim.mode == 'api':
    import run_API as ask
elif config.sim.mode == 'gpu':
    required_local_fields = ["model_name", "API_TOKEN", "quantized"]
    missing_local_fields = [field for field in required_local_fields if not hasattr(config.model, field)]
    if missing_local_fields:
        raise ValueError(f"sim.mode 为 'gpu' 时需要在 config.yaml 的 model 下配置: {missing_local_fields}。当前配置已精简为 API 模式，请使用 sim.mode: 'api'。")
    import run_local as ask
else:
    raise ValueError(f"Unsupported sim.mode: {config.sim.mode}")


def _answer_for_task(task):
    if task[0] == "fixed":
        return task[1]
    _, prompt, opts = task
    return ask.get_response(prompt, options=opts)


def _parallel_answers(tasks):
    if len(tasks) == 1:
        return [_answer_for_task(tasks[0])]
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        return list(ex.map(_answer_for_task, tasks))


def _bias_batch_size():
    try:
        ap = config.api.active_provider
        mc = getattr(config.api.providers[ap], "max_concurrency", None)
        if mc:
            return max(1, int(mc))
    except Exception:
        pass
    return 8


def simulate_meta_prompting(memory_size, rewards, options, fname):
    question_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
    try:
        tracker = pickle.load(open(fname, 'rb'))
    except:
        tracker = {q: [] for q in question_list}

    new_options = options.copy()

    while len(tracker[question_list[0]])<100:
        t = len(tracker[question_list[0]])
        random.shuffle(new_options)
        rules = pr.get_rules(rewards, options = new_options)

        running_player = mp.running_player(options = new_options, memory_size=memory_size, rewards=rewards)

        i, questions, q_list, prompts = mp.get_meta_prompt_list(some_player = running_player, rules=rules, options=new_options)


        for prompt, question, q in zip(prompts, questions, q_list):


            response = ask.get_meta_response(prompt)
            gold_response = mp.gold_sim(q, question, running_player, i, options)

            if q == 'actions':
                if all(option in response for option in options):
                    tracker[q].append(1)
                    print('Success')
                else:
                    tracker[q].append(0)
            else:
                print("GOLD: ", gold_response) 
                if gold_response in response:
                    tracker[q].append(1)
                    print('SUCCESS')
                else:
                    tracker[q].append(0)

        print(f"INTERACTION {t}")
        if t % 5 == 0:
            ut.save_pickle(tracker, fname)
    return tracker


def individual(dataframe, memory_size, rewards, options, fname, repeats):
    new_options = options.copy()
    player = dataframe['simulation']
    ut.compact_individual_dataframe(dataframe, options)
    tracker = dataframe['tracker']
    batch = _bias_batch_size()
    while len(tracker['answers']) < repeats:


        need = repeats - len(tracker['answers'])
        _tasks = []
        for _ in range(min(batch, need)):
            random.shuffle(new_options)
            rules = pr.get_rules(rewards, options = new_options)

            prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)
            _tasks.append(("llm", prompt, list(new_options)))


        tracker['answers'].extend(_parallel_answers(_tasks))

        n = len(tracker['answers'])
        _emit_progress(min(n, repeats), repeats)
        dataframe['tracker'] = tracker
        ut.save_pickle(dataframe, fname)

    dataframe['tracker'] = tracker


    _emit_progress(min(len(tracker['answers']), repeats), repeats)


def population(dataframe, run, memory_size, rewards, options, fname, runs=1):
    new_options = options.copy()
    interaction_dict = dataframe['simulation']
    tracker = dataframe['tracker']

    progress_cap = POPULATION_PROGRESS_CAP

    def _pop_progress(interactions: int, run_done: bool = False) -> None:
        seg = progress_cap if run_done else min(interactions, progress_cap)
        _report_run_progress(run, seg, progress_cap, max(1, runs))


    max_interactions = ut.convergence_cap()
    _pop_progress(0)
    while ut.has_tracker_converged(tracker) == False:
        if max_interactions and len(tracker['outcome']) >= max_interactions:
            print(f"[WARN] 本 run 在 {max_interactions} 次交互内未达成收敛"
                  f"（最近 {config.params.convergence_time} 次未全胜），触发硬上限并停止；按未收敛记录。"
                  " 如需更长演化请调大 config 的 params.convergence_max_interactions（设 0 取消上限）。",
                  flush=True)
            break

        p1 = random.choice(list(interaction_dict.keys()))
        p2 = random.choice(interaction_dict[p1]['neighbours'])


        interaction_dict[p1]['interactions'].append(p2)
        interaction_dict[p2]['interactions'].append(p1)
        p1_dict = interaction_dict[p1]
        p2_dict = interaction_dict[p2]


        _tasks = []
        for player in [p1_dict, p2_dict]:
            random.shuffle(new_options)
            rules = pr.get_rules(rewards, options = new_options)

            prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)
            _tasks.append(("llm", prompt, list(new_options)))


        answers = _parallel_answers(_tasks)

        my_answer, partner_answer = answers


        outcome = ut.get_outcome(my_answer, partner_answer, rewards)
        interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
        interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
        ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)

        if len(tracker['outcome']) % 50 == 0:
            _pop_progress(len(tracker['outcome']))
            dataframe['simulation'] = interaction_dict
            dataframe['tracker'] = tracker
            ut.save_pickle(dataframe, fname)

    _pop_progress(len(tracker['outcome']), run_done=True)
    dataframe['simulation'] = interaction_dict
    dataframe['tracker'] = tracker
    dataframe['convergence'] = {'converged_index': len(tracker['outcome']), 'committed_to': None}


def committed(dataframe, run, memory_size, rewards, options, fname, total_interactions=total_interactions, runs=1):
    new_options = options.copy()
    interaction_dict = dataframe['simulation']
    tracker = dataframe['tracker']
    init_tracker_len = dataframe['convergence']['converged_index']


    _p = getattr(ut, "config", None)
    _p = getattr(_p, "params", None)
    _early_stop_on = bool(getattr(_p, "committed_early_stop", False))
    _es_flip = float(getattr(_p, "early_stop_flip_threshold", 0.95))
    _es_window = int(getattr(_p, "early_stop_window", 72))
    _es_noflip_frac = float(getattr(_p, "early_stop_noflip_frac", 0.5))
    _es_noflip_thr = float(getattr(_p, "early_stop_noflip_threshold", 0.10))
    _nc_answers = []

    def _committed_progress(done: int, run_done: bool = False) -> None:
        seg = total_interactions if run_done else done
        _report_run_progress(run, seg, total_interactions, max(1, runs))

    _committed_progress(0)
    while len(tracker['outcome']) - init_tracker_len < total_interactions:
        random.shuffle(new_options)
        rules = pr.get_rules(rewards, options = new_options)


        p1 = random.choice(list(interaction_dict.keys()))
        p2 = random.choice(interaction_dict[p1]['neighbours'])


        interaction_dict[p1]['interactions'].append(p2)
        interaction_dict[p2]['interactions'].append(p1)
        p1_dict = interaction_dict[p1]
        p2_dict = interaction_dict[p2]


        _opts = list(new_options)
        _tasks = []
        for player in [p1_dict, p2_dict]:

            if player['committed_tag'] == True:
                _tasks.append(("fixed", dataframe['convergence']['committed_to']))
            else:

                prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)
                _tasks.append(("llm", prompt, _opts))


        answers = _parallel_answers(_tasks)

        my_answer, partner_answer = answers


        outcome = ut.get_outcome(my_answer, partner_answer, rewards)
        interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
        interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
        ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)

        done = len(tracker['outcome']) - init_tracker_len
        if _early_stop_on:
            for _pd, _ans in ((p1_dict, my_answer), (p2_dict, partner_answer)):
                if _pd['committed_tag'] != True:
                    _nc_answers.append(_ans)
            if len(_nc_answers) >= _es_window:
                _adopt = sum(1 for a in _nc_answers[-_es_window:]
                             if a == dataframe['convergence']['committed_to']) / _es_window
                _stop_reason = None
                if _adopt >= _es_flip:
                    _stop_reason = 'flip'
                elif done >= total_interactions * _es_noflip_frac and _adopt <= _es_noflip_thr:
                    _stop_reason = 'no_flip'
                if _stop_reason:
                    dataframe['convergence']['early_stop'] = {
                        'reason': _stop_reason, 'at_interaction': done,
                        'adoption_last_window': round(_adopt, 4), 'window': _es_window}
                    print(f"[early-stop] run {run}: 非坚定者近 {_es_window} 个答案采纳率 "
                          f"{_adopt:.0%} → 判 {_stop_reason}，于 {done}/{total_interactions} 次交互提前结束")
                    break
        if done % 20 == 0:
            print(fname)
            _committed_progress(done)
            dataframe['simulation'] = interaction_dict
            dataframe['tracker'] = tracker
            ut.save_pickle(dataframe, fname)

    _committed_progress(total_interactions, run_done=True)
    dataframe['simulation'] = interaction_dict
    dataframe['tracker'] = tracker

