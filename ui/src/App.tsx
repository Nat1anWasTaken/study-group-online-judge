import { useEffect, useState } from "react";
import {
    ArrowDownRight,
    ArrowUpRight,
    ArrowUpRight as External,
    Check,
    RefreshCw,
    Search,
    Trophy,
} from "lucide-react";
import "./styles.css";
import ceruleanLogo from "./assets/cerulean.png";
import labNames from "./lab-names.json";
import greetings from "./greetings.json";

type Entry = {
    rank: number;
    github_actor: string;
    score: number | null;
    submitted_at: string;
    run_url: string | null;
    submission_id: string;
    attempts: number;
    participant_id?: string;
    val_loss?: number;
};
type Lab = {
    id: string;
    grading_type: "pass_fail" | "score";
    primary_metric: string | null;
    metric_direction: string | null;
    entries: Entry[];
    submissions: number;
    participants: number;
    project_url?: string | null;
};
type Snapshot = {
    source: string;
    updated_at: string | null;
    error: string | null;
    labs: Lab[];
};
const names: Record<string, string> = labNames;
const date = (value: string) =>
    new Date(value).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
function runLink(url: string | null) {
    try {
        return url && new URL(url).protocol === "https:" ? url : null;
    } catch {
        return null;
    }
}

function ParticipantAvatar({ username, flock }: { username: string; flock: boolean }) {
    const [failed, setFailed] = useState(false);
    return (
        <span className="avatar" aria-hidden="true">
            {failed || flock ? (
                username.slice(0, 2).toUpperCase()
            ) : (
                <img
                    src={`https://github.com/${encodeURIComponent(username)}.png?size=72`}
                    alt=""
                    width={36}
                    height={36}
                    loading="lazy"
                    decoding="async"
                    referrerPolicy="no-referrer"
                    onError={() => setFailed(true)}
                />
            )}
        </span>
    );
}

export default function App() {
    const [greeting] = useState(
        () => greetings[Math.floor(Math.random() * greetings.length)],
    );
    const [data, setData] = useState<Snapshot | null>(null);
    const [selected, setSelected] = useState("");
    const [query, setQuery] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    useEffect(() => {
        const controller = new AbortController();
        const load = async () => {
            setLoading(true);
            try {
                const response = await fetch("/api/leaderboard", {
                    signal: controller.signal,
                });
                if (!response.ok)
                    throw new Error(
                        "Leaderboard unavailable. Please try again.",
                    );
                const snapshot: Snapshot = await response.json();
                snapshot.labs.sort((a, b) => {
                    if (a.id === "midterm") return -1;
                    if (b.id === "midterm") return 1;
                    return b.id.localeCompare(a.id, undefined, { numeric: true });
                });
                setData(snapshot);
                setSelected(
                    (previous) => snapshot.labs.some((item) => item.id === previous)
                        ? previous : snapshot.labs[0]?.id || "",
                );
                setError("");
            } catch (e) {
                if (!controller.signal.aborted)
                    setError(
                        e instanceof Error
                            ? e.message
                            : "Unable to load standings.",
                    );
            } finally {
                if (!controller.signal.aborted) setLoading(false);
            }
        };
        void load();
        const interval = setInterval(() => void load(), 30_000);
        return () => {
            controller.abort();
            clearInterval(interval);
        };
    }, []);
    const lab = data?.labs.find((item) => item.id === selected);
    const midterm = lab?.id === "midterm";
    const scored = lab?.grading_type === "score";
    const entries =
        lab?.entries.filter((entry) =>
            entry.github_actor.toLowerCase().includes(query.toLowerCase()),
        ) ?? [];
    return (
        <div className="shell">
            <header className="nav">
                <a href="/" className="brand">
                    <span className="brand-mark">
                        <img src={ceruleanLogo} alt="" />
                    </span>{" "}
                    Cerulean Labs
                </a>
                {runLink(lab?.project_url ?? null) && (
                    <a
                        className="nav-link"
                        href={runLink(lab?.project_url ?? null)!}
                        target="_blank"
                        rel="noreferrer"
                    >
                        View experiments <External size={16} />
                    </a>
                )}
            </header>
            <main>
                <section className="hero">
                    <h1>{greeting}</h1>
                </section>
                <div className="workspace">
                    <div className="lab-toolbar" id="standings">
                        <aside aria-label="Choose a leaderboard">
                            {data?.labs.map((item) => (
                                <button
                                    key={item.id}
                                    aria-pressed={selected === item.id}
                                    className={`lab-button ${selected === item.id ? "active" : ""}`}
                                    onClick={() => {
                                        setSelected(item.id);
                                        setQuery("");
                                    }}
                                >
                                    <span className="lab-number">
                                        {item.id === "midterm" ? "MT" : item.id.replace("lab", "").padStart(2, "0")}
                                    </span>
                                    <strong>
                                        {item.id === "midterm" ? "Midterm" : item.id.replace("lab", "Lab ")}
                                    </strong>
                                    <small>
                                        {names[item.id] ??
                                            "Study group experiment"}
                                    </small>
                                </button>
                            ))}
                        </aside>
                    </div>
                    <section className="board" aria-label="Leaderboard">
                        <div className="board-header">
                            <div>
                                <div className="eyebrow accent">
                                    {midterm ? "MIDTERM · TRACK 3" : lab?.id.replace("lab", "LAB ") ?? "STANDINGS"}
                                </div>
                                <h2>
                                    {lab
                                        ? (names[lab.id] ?? lab.id)
                                        : "Loading experiments"}
                                </h2>
                                {(midterm || !scored) && (
                                    <p>
                                        {midterm
                                            ? "Fewest training steps with final validation loss < 3.28. Best qualifying run per participant."
                                            : "Correctness first. Ranked by earliest passing submission."}
                                    </p>
                                )}
                            </div>
                            <div className="board-status">
                                <span className="mode">
                                    {scored ? (
                                        lab?.metric_direction === "maximize" ? (
                                            <ArrowUpRight size={14} />
                                        ) : (
                                            <ArrowDownRight size={14} />
                                        )
                                    ) : (
                                        <Check size={14} />
                                    )}{" "}
                                    {midterm ? "Optimization speedrun" : scored ? "Score challenge" : "Pass / fail"}
                                </span>
                                <span
                                    className="sync"
                                    title={data?.updated_at ? date(data.updated_at) : undefined}
                                >
                                    <RefreshCw
                                        size={13}
                                        className={loading ? "spinning" : ""}
                                    />
                                    {data?.updated_at
                                        ? `Updated from Flock at ${new Date(data.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
                                        : "Waiting for Flock"}
                                </span>
                            </div>
                        </div>
                        {(error || data?.error) && (
                            <div className="error" role="alert">
                                {error || data?.error}
                            </div>
                        )}
                        <div className="table-toolbar">
                            <h3>
                                Standings{" "}
                                <span>{lab?.entries.length ?? 0}</span>
                            </h3>
                            <label className="search">
                                <Search size={15} />
                                <input
                                    aria-label="Search participants"
                                    placeholder="Find a participant…"
                                    value={query}
                                    onChange={(event) =>
                                        setQuery(event.target.value)
                                    }
                                />
                            </label>
                        </div>
                        <div className="table-scroll">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Participant</th>
                                        <th>
                                            {midterm ? "Training steps ↓" : scored
                                                ? `${lab?.primary_metric} ${lab?.metric_direction === "maximize" ? "↑" : "↓"}`
                                                : "Result"}
                                        </th>
                                        {midterm && <th>Validation loss</th>}
                                        <th>{midterm ? "Run started" : "Submitted"}</th>
                                        <th>
                                            <span className="sr-only">Run</span>
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {entries.map((entry) => (
                                        <tr key={entry.participant_id ?? entry.github_actor}>
                                            <td>
                                                <span
                                                    className={
                                                        entry.rank === 1
                                                            ? "rank first"
                                                            : "rank"
                                                    }
                                                >
                                                    {String(
                                                        entry.rank,
                                                    ).padStart(2, "0")}
                                                </span>
                                            </td>
                                            <td>
                                                <div className="participant">
                                                    <ParticipantAvatar
                                                        key={entry.github_actor}
                                                        flock={entry.participant_id?.startsWith("flock:") ?? false}
                                                        username={
                                                            entry.github_actor
                                                        }
                                                    />
                                                    <div>
                                                        <strong>
                                                            {entry.github_actor}
                                                        </strong>
                                                        <small>
                                                            {entry.attempts}{" "}
                                                            {midterm
                                                                ? entry.attempts === 1 ? "run" : "runs"
                                                                : entry.attempts === 1 ? "submission" : "submissions"}
                                                        </small>
                                                    </div>
                                                </div>
                                            </td>
                                            <td>
                                                {scored ? (
                                                    <span className="score">
                                                        {entry.score?.toLocaleString(
                                                            undefined,
                                                            {
                                                                maximumSignificantDigits: 7,
                                                            },
                                                        )}
                                                    </span>
                                                ) : (
                                                    <span className="passed">
                                                        <Check size={12} />{" "}
                                                        Passed
                                                    </span>
                                                )}
                                            </td>
                                            {midterm && <td className="score">{entry.val_loss?.toLocaleString(undefined, { maximumFractionDigits: 5 })}</td>}
                                            <td className="submitted">
                                                {date(entry.submitted_at)}
                                            </td>
                                            <td>
                                                {runLink(entry.run_url) && (
                                                    <a
                                                        className="run-link"
                                                        href={runLink(
                                                            entry.run_url,
                                                        )!}
                                                        target="_blank"
                                                        rel="noreferrer"
                                                        aria-label={`View ${entry.github_actor}'s Flock run`}
                                                    >
                                                        <External size={16} />
                                                    </a>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {!entries.length && (
                            <div className="empty">
                                <Trophy size={26} />
                                <h3>
                                    {query
                                        ? "No matching participants"
                                        : loading && !data
                                          ? "Loading standings…"
                                          : "The next result could be yours."}
                                </h3>
                                <p>
                                    {query
                                        ? "Try another participant name."
                                        : "Completed, qualifying Flock runs will appear here."}
                                </p>
                            </div>
                        )}
                    </section>
                </div>
                <footer>
                    <span>
                        <span className="brand-name">Cerulean Labs</span>
                        <span className="footer-muted">/ AI for everyone.</span>
                    </span>
                </footer>
            </main>
        </div>
    );
}
