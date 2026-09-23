CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    repo_url TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    task_id TEXT NOT NULL,
    github_actor TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    result_json TEXT,
    error TEXT,
    wandb_run_id TEXT,
    wandb_url TEXT
);

CREATE INDEX jobs_status_created_at ON jobs (status, created_at);
