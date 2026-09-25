CREATE TABLE remote_events (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    sequence INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    PRIMARY KEY (job_id, sequence)
);

CREATE TABLE remote_report_cursor (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id),
    last_sequence INTEGER NOT NULL DEFAULT 0
);
