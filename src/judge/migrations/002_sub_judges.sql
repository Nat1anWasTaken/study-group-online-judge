CREATE TABLE sub_judges (
    id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    task_ids_json TEXT NOT NULL,
    max_gpus INTEGER NOT NULL,
    judge_revision TEXT NOT NULL,
    registered_at TEXT NOT NULL
);

ALTER TABLE jobs ADD COLUMN assigned_judge_id TEXT;
ALTER TABLE jobs ADD COLUMN slurm_job_id TEXT;
ALTER TABLE jobs ADD COLUMN request_key TEXT;

CREATE UNIQUE INDEX jobs_repo_request_key
ON jobs (repo_url, request_key)
WHERE request_key IS NOT NULL;

CREATE INDEX jobs_assignment_status_created_at
ON jobs (assigned_judge_id, status, created_at);
