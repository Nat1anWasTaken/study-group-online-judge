import json
import math
import os
import re
import subprocess
from pathlib import Path

import wandb

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/work/nat1andotxyz/lab4/dedup-s')
VALIDATION_HASH = 'eb90498c391ae42ffa10785ab39c5f9abc1b198068418ce3bf7655caefb6151a'


def git(path, *arguments):
    return subprocess.check_output(['git', '-C', str(path), *arguments], text=True).strip()


def choose(api, candidates):
    evaluated = []
    for candidate in candidates:
        runs = list(api.runs('cerulean-labs/gpt2-training',
                            filters={'config.slurm_job_id': candidate['training_job']}))
        if len(runs) != 1:
            raise RuntimeError(f"Expected one run for {candidate['experiment']}")
        run = runs[0]
        summary = run.summary
        config = run.config
        ppl = summary.get('final_eval_perplexity')
        if not summary.get('evaluation_completed') or not summary.get('training_completed'):
            raise RuntimeError(f'{run.id}: final evaluation is incomplete')
        if ppl is None or not math.isfinite(ppl):
            raise RuntimeError(f'{run.id}: invalid final PPL')
        if config.get('git_commit') != candidate['commit']:
            raise RuntimeError(f'{run.id}: source commit mismatch')
        if config.get('training_budget_seconds') != 1620:
            raise RuntimeError(f'{run.id}: training budget mismatch')
        if config.get('validation', {}).get('token_ids_sha256') != VALIDATION_HASH:
            raise RuntimeError(f'{run.id}: validation mismatch')
        if config.get('muon_learning_rate') != candidate['learning_rate']:
            raise RuntimeError(f'{run.id}: learning rate mismatch')
        evaluated.append(dict(candidate, run_id=run.id, final_eval_perplexity=ppl))
    winner = min(evaluated, key=lambda row: row['final_eval_perplexity'])
    return dict(criterion='minimum final dev perplexity; same 2048-document validation',
                winner=winner, candidates=evaluated)


def launch():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Launch selection requires a Slurm allocation')
    ready = json.loads((ROOT / 'ready.json').read_text())
    for method in ('control', 'exact', 'minhash'):
        if ready['methods'][method]['blocks'] != 1_228_800:
            raise RuntimeError(f'{method}: incomplete preparation')
    api = wandb.Api()
    candidates = json.loads((SOURCE / 'dedup_candidates.json').read_text())
    selections = {schedule: choose(api, rows) for schedule, rows in candidates.items()}
    (ROOT / 'selection.json').write_text(json.dumps(selections, indent=2))
    experiments = json.loads((SOURCE / 'dedup_experiments.json').read_text())
    submissions = []
    environment = dict(os.environ, SBATCH_ACCOUNT='ACD115198')
    for experiment in experiments:
        path = Path(experiment['worktree'])
        if git(path, 'status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError(f'{path}: tracked files changed before launch')
        selection = selections[experiment['schedule']]
        source = path / 'src/labs/train_lab4.py'
        text, replacements = re.subn(r'^LEARNING_RATE = .*$',
                                    f"LEARNING_RATE = {selection['winner']['learning_rate']!r}",
                                    source.read_text(), flags=re.M)
        if replacements != 1:
            raise RuntimeError(f'{path}: missing learning rate constant')
        source.write_text(text)
        recipe = dict(selection, experiment=experiment['experiment'],
                      method=experiment['method'], schedule=experiment['schedule'],
                      preparation_commit=ready['preparation_commit'],
                      preparation_job=ready['preparation_job'], launch_job=os.environ['SLURM_JOB_ID'])
        (path / 'src/labs/dedup_recipe.json').write_text(json.dumps(recipe, indent=2)+'\n')
        git(path, 'add', 'src/labs/train_lab4.py', 'src/labs/dedup_recipe.json')
        git(path, 'commit', '-m', f"lab4 {experiment['experiment']}: select best {experiment['schedule']} learning rate")
        output = subprocess.check_output(['bash', 'src/labs/submit_lab4.sh'], cwd=path,
                                         env=environment, text=True)
        print(output, flush=True)
        training = re.search(r'Training job: (\d+)', output)
        publish = re.search(r'Evaluation/publish job: (\d+)', output)
        if not training or not publish:
            raise RuntimeError(f'{path}: incomplete submission output: {output}')
        submissions.append(dict(experiment, learning_rate=selection['winner']['learning_rate'],
                                control_run=selection['winner']['run_id'], commit=git(path, 'rev-parse', 'HEAD'),
                                training_job=training.group(1), publish_job=publish.group(1)))
        (ROOT / 'submissions.json').write_text(json.dumps(submissions, indent=2))
    print(json.dumps(submissions, indent=2), flush=True)


if __name__ == '__main__':
    launch()
