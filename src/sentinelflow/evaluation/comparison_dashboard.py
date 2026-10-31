"""Interactive comparison dashboard for SentinelFlow concept-drift evaluations."""

# ruff: noqa: E501
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "static_logistic": "Static Logistic",
    "static_random_forest": "Static Random Forest",
    "adaptive_none": "Adaptive None",
    "adaptive_adwin": "Adaptive ADWIN",
    "adaptive_ddm": "Adaptive DDM",
    "online_ensemble": "Online Ensemble",
}

MODEL_ORDER = list(DISPLAY_NAMES)
PHASES = ("pre_drift", "ramp_drift", "drifted")

METRICS = (
    "precision",
    "recall",
    "f1",
    "fpr",
    "roc_auc",
    "average_precision",
    "precision_at_100",
)

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "fpr": "FPR",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision",
    "precision_at_100": "Precision@100",
}

METRIC_SHORT_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "fpr": "FPR",
    "roc_auc": "ROC-AUC",
    "average_precision": "Avg. Precision",
    "precision_at_100": "P@100",
}

METRIC_EXPLANATIONS = {
    "precision": (
        "Among transactions predicted as fraud, the proportion that are actually fraudulent."
    ),
    "recall": ("Among actual fraud transactions, the proportion detected by the model."),
    "f1": "The harmonic mean of Precision and Recall.",
    "fpr": ("The proportion of legitimate transactions incorrectly flagged as fraud."),
    "roc_auc": (
        "How well the model ranks fraud above legitimate transactions across "
        "classification thresholds."
    ),
    "average_precision": (
        "A precision-recall ranking summary that is useful for imbalanced fraud-detection data."
    ),
    "precision_at_100": ("The proportion of true fraud among the 100 highest-risk predictions."),
}

# Lexend (Google Fonts) with system fallbacks. Used for the page, the SVG
# and every Plotly chart.
FONT_STACK = '"Lexend", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif'
GOOGLE_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Lexend:'
    'wght@400;500;600;700&display=swap" rel="stylesheet">'
)

# Single source of truth for colour. Keys become CSS custom properties
# (underscores -> hyphens) and are also passed to the Plotly charts.
PALETTE = {
    # Neutrals
    "ink": "#1d1d1f",
    "muted": "#6e6e73",
    "bg": "#f5f5f7",
    "surface": "#ffffff",
    "line": "#d2d2d7",
    "line_soft": "#e5e5ea",
    "plot_bg": "#f2f2f5",
    # Header and context panels: blue (teal-blue)
    "accent": "#0e7c86",
    # Recovery Result card: purple
    "recovery": "#6b3fa0",
    "recovery_tint": "#f5effb",
    "recovery_border": "#cdb6e6",
    # Initial evaluation: yellow (initial_ink is the darker, text-safe shade)
    "initial": "#f2b705",
    "initial_ink": "#7a5800",
    "initial_tint": "#fff9db",
    "initial_border": "#f1d96a",
    # Revised evaluation: green
    "revised": "#16803a",
    "revised_tint": "#eefaf1",
    "revised_border": "#a8dbb7",
    # Semantic colours
    "good": "#2e7d32",
    "bad": "#c0261c",
    # Model identity colours
    "static": "#6e6e73",
    # Eval node highlight
    "focus_fill": "#fff3bf",
    "focus_hover": "#ffe680",
}

# One colour per drift phase, shared by both evaluation sections.
PHASE_COLORS = ["#4e79a7", "#f28e2b", "#8e6bbf"]  # pre, ramp, fully drifted

# Details that are not part of the detector-quality CSV.
DETECTOR_DETAILS = {
    "ADWIN": {
        "label": "Adaptive ADWIN",
        "source_batch": 17,
        "observed_batch": 24,
        "delay": "7 batches",
        "window": "200,000 recent labeled rows",
    },
    "DDM": {
        "label": "Adaptive DDM",
        "source_batch": 17,
        "observed_batch": 24,
        "delay": "7 batches",
        "window": "200,000 recent labeled rows",
    },
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _display_name(model: str) -> str:
    """Return a human-readable model name."""
    return DISPLAY_NAMES.get(model, model)


def _normalize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy metric column names."""
    normalized = frame.copy()

    if "average_precision" not in normalized.columns and "pr_auc" in normalized.columns:
        normalized = normalized.rename(columns={"pr_auc": "average_precision"})

    return normalized


def _ordered_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort models using the project display order."""
    ordered = frame.copy()

    ordered["model_order"] = ordered["model"].map(
        {model: index for index, model in enumerate(MODEL_ORDER)}
    )
    ordered["display_model"] = ordered["model"].map(_display_name)

    return ordered.sort_values("model_order")


def _require_columns(name: str, frame: pd.DataFrame, required: set[str]) -> None:
    """Raise a readable error when a summary is missing columns."""
    missing = required.difference(frame.columns)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{name} summary is missing required columns: {names}")


def _with_font(html: str) -> str:
    """Inject the Lexend font links into a Plotly-generated page."""
    return html.replace("<head>", "<head>" + GOOGLE_FONT_LINKS, 1)


def _write_text(path: Path, content: str) -> Path:
    """Write text to ``path``, creating parent directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _phase_metric_data(
    frame: pd.DataFrame,
    metric: str,
) -> dict[str, object]:
    """Return metric values for all three drift phases."""
    phase_values: dict[str, dict[str, float]] = {}

    for phase in PHASES:
        phase_frame = _ordered_models(frame[frame["phase"] == phase])

        phase_values[phase] = {
            row["model"]: float(row[metric]) for _, row in phase_frame.iterrows()
        }

    models = [
        model for model in MODEL_ORDER if all(model in phase_values[phase] for phase in PHASES)
    ]

    return {
        "models": models,
        "labels": [_display_name(model) for model in models],
        "pre": [phase_values["pre_drift"][model] for model in models],
        "ramp": [phase_values["ramp_drift"][model] for model in models],
        "drifted": [phase_values["drifted"][model] for model in models],
    }


def _relative_change_data(
    frame: pd.DataFrame,
    metric: str,
) -> dict[str, object]:
    """Return relative change from pre-drift to fully drifted.

    Positive values are always "worse": a drop for most metrics, a rise
    for FPR.
    """
    phase_data = _phase_metric_data(frame, metric)
    values = []

    for pre, drifted in zip(
        phase_data["pre"],
        phase_data["drifted"],
        strict=True,
    ):
        if pre == 0:
            values.append(0.0)
            continue

        change = (drifted - pre) if metric == "fpr" else (pre - drifted)
        values.append(change / pre * 100.0)

    return {
        "models": phase_data["models"],
        "labels": phase_data["labels"],
        "values": values,
    }


def _phase_value(
    frame: pd.DataFrame,
    model: str,
    phase: str,
    metric: str = "recall",
) -> float:
    """Return one model/phase metric value."""
    rows = frame[(frame["model"] == model) & (frame["phase"] == phase)]

    if rows.empty:
        raise ValueError(f"Missing {model}/{phase}/{metric} result.")

    return float(rows.iloc[0][metric])


# ---------------------------------------------------------------------------
# Legacy dashboards (unchanged behaviour)
# ---------------------------------------------------------------------------


def _build_legacy_comparison_chart(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    metric: str,
    first_name: str,
    second_name: str,
) -> go.Figure:
    """Build the original standalone comparison chart."""
    figure = go.Figure()

    for name, frame, color in (
        (first_name, first, PALETTE["initial"]),
        (second_name, second, PALETTE["revised"]),
    ):
        ordered = _ordered_models(frame)

        figure.add_trace(
            go.Bar(
                name=name,
                x=ordered["display_model"],
                y=ordered[metric],
                marker_color=color,
            )
        )

    figure.update_layout(
        barmode="group",
        title=f"{first_name} vs. {second_name}",
        yaxis_title=METRIC_LABELS.get(metric, metric),
        font={"family": FONT_STACK, "color": PALETTE["ink"]},
        hoverlabel={"font": {"family": FONT_STACK, "size": 10}},
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["plot_bg"],
    )

    return figure


def generate_scenario_comparison_dashboard(
    phase_one_summary: Path,
    phase_two_summary: Path,
    output_path: Path,
    *,
    phase_one_name: str = "Original Drift",
    phase_two_name: str = "Camouflage Drift",
    metric: str = "recall",
) -> Path:
    """Generate the legacy scenario-comparison dashboard."""
    phase_one = _normalize_metrics(pd.read_csv(phase_one_summary))
    phase_two = _normalize_metrics(pd.read_csv(phase_two_summary))

    required = {"model", metric}
    _require_columns(phase_one_name, phase_one, required)
    _require_columns(phase_two_name, phase_two, required)

    figure = _build_legacy_comparison_chart(
        phase_one,
        phase_two,
        metric=metric,
        first_name=phase_one_name,
        second_name=phase_two_name,
    )

    return _write_text(
        output_path,
        _with_font(pio.to_html(figure, full_html=True, include_plotlyjs=True)),
    )


def generate_phase_comparison_dashboard(
    original_phase_summary: Path,
    camouflage_phase_summary: Path,
    output_path: Path,
    *,
    metric: str = "recall",
) -> Path:
    """Generate the legacy phase-comparison dashboard."""
    original = _normalize_metrics(pd.read_csv(original_phase_summary))
    camouflage = _normalize_metrics(pd.read_csv(camouflage_phase_summary))

    figure = go.Figure()

    for scenario, frame, pattern in (
        ("Original Drift", original, ""),
        ("Camouflage Drift", camouflage, "/"),
    ):
        for phase, color in zip(PHASES, PHASE_COLORS, strict=True):
            ordered = _ordered_models(frame[frame["phase"] == phase])

            figure.add_trace(
                go.Bar(
                    name=f"{scenario} - {phase.replace('_', ' ').title()}",
                    x=ordered["display_model"],
                    y=ordered[metric],
                    marker={"color": color, "pattern": {"shape": pattern}},
                )
            )

    figure.update_layout(
        barmode="group",
        title="Performance Across Drift Phases",
        yaxis_title=METRIC_LABELS.get(metric, metric),
        font={"family": FONT_STACK, "color": PALETTE["ink"]},
        hoverlabel={"font": {"family": FONT_STACK, "size": 10}},
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["plot_bg"],
    )

    return _write_text(
        output_path,
        _with_font(pio.to_html(figure, full_html=True, include_plotlyjs=True)),
    )


# ---------------------------------------------------------------------------
# Final dashboard: HTML fragment builders
# ---------------------------------------------------------------------------


def _root_css_vars() -> str:
    """Render the palette and font as CSS custom properties."""
    lines = [f"--{key.replace('_', '-')}: {value};" for key, value in PALETTE.items()]
    lines.append(f"--font: {FONT_STACK};")
    return "\n    ".join(lines)


def _metric_buttons_html() -> str:
    return "\n".join(
        "<button "
        f'class="metric-button{" active" if metric == "recall" else ""}" '
        'type="button" '
        f'data-metric="{metric}">'
        f"{escape(METRIC_SHORT_LABELS[metric])}"
        "</button>"
        for metric in METRICS
    )


def _trio(values: dict[str, float]) -> str:
    """Format a pre -> ramp -> drifted triple."""
    return f"{values['pre']:.3f} → {values['ramp']:.3f} → {values['drifted']:.3f}"


def _comparison_table_html(
    rows: list[tuple[str, str, dict[str, float], str]],
) -> str:
    """Render the pre/ramp/drifted table.

    ``rows`` items are ``(label, css_class, values, final_cell_class)``.
    """
    cells = [
        f'<div class="comparison-cell header">{title}</div>'
        for title in ("Model", "Pre-drift", "Ramp drift", "Fully drifted")
    ]

    for label, css_class, values, final_class in rows:
        cells.append(f'<div class="comparison-cell model {css_class}">{escape(label)}</div>')
        cells.append(f'<div class="comparison-cell value">{values["pre"]:.3f}</div>')
        cells.append(f'<div class="comparison-cell value">{values["ramp"]:.3f}</div>')
        cells.append(
            f'<div class="comparison-cell value {final_class}">{values["drifted"]:.3f}</div>'
        )

    return (
        '<div class="table-metric">Recall</div>\n'
        '<div class="model-comparison-table">\n' + "\n".join(cells) + "\n</div>"
    )


def _detector_card_html(detector: str, alarms: object, false_alarms: object) -> str:
    details = DETECTOR_DETAILS[detector]

    fields = (
        ("Concept-drift detector", detector),
        ("Confirmed alarms", alarms),
        ("False alarms", false_alarms),
        ("Source batch", details["source_batch"]),
        ("Observed batch", details["observed_batch"]),
        ("Detection delay", details["delay"]),
        ("Retraining window", details["window"]),
    )

    body = "\n".join(
        f"<p><strong>{escape(name)}:</strong> {escape(str(value))}</p>" for name, value in fields
    )

    return f'<div class="detector-card"><h3>{escape(details["label"])}</h3>\n{body}\n</div>'


# ---------------------------------------------------------------------------
# Final dashboard
# ---------------------------------------------------------------------------


def generate_final_experiment_dashboard(
    separability_batch_metrics: Path,
    separability_phase_summary: Path,
    camouflage_batch_metrics: Path,
    camouflage_phase_summary: Path,
    detector_quality_path: Path,
    output_path: Path,
) -> Path:
    """Generate the compact SentinelFlow evaluation dashboard."""
    # Batch metrics are only checked for existence/readability here.
    pd.read_csv(separability_batch_metrics)
    pd.read_csv(camouflage_batch_metrics)

    initial = _normalize_metrics(pd.read_csv(separability_phase_summary))
    revised = _normalize_metrics(pd.read_csv(camouflage_phase_summary))
    detector_quality = pd.read_csv(detector_quality_path)

    required = {"model", "phase", *METRICS}
    _require_columns("Initial evaluation", initial, required)
    _require_columns("Revised evaluation", revised, required)

    dashboard_data: dict[str, object] = {
        "metricLabels": METRIC_LABELS,
        "metricExplanations": METRIC_EXPLANATIONS,
        "palette": PALETTE,
        "phaseColors": PHASE_COLORS,
        "font": FONT_STACK,
        "initial": {},
        "revised": {},
    }

    for key, frame in (("initial", initial), ("revised", revised)):
        for metric in METRICS:
            dashboard_data[key][metric] = {
                "performance": _phase_metric_data(frame, metric),
                "loss": _relative_change_data(frame, metric),
            }

    comparison_models = (
        "static_random_forest",
        "adaptive_adwin",
        "adaptive_ddm",
    )

    values: dict[str, dict[str, dict[str, float]]] = {}

    for key, frame in (("initial", initial), ("revised", revised)):
        values[key] = {
            model: {
                "pre": _phase_value(frame, model, "pre_drift"),
                "ramp": _phase_value(frame, model, "ramp_drift"),
                "drifted": _phase_value(frame, model, "drifted"),
            }
            for model in comparison_models
        }

    def f1_drifted(model: str) -> float:
        return _phase_value(revised, model, "drifted", "f1")

    initial_rf = values["initial"]["static_random_forest"]
    initial_adwin = values["initial"]["adaptive_adwin"]
    initial_ddm = values["initial"]["adaptive_ddm"]
    revised_rf = values["revised"]["static_random_forest"]
    revised_adwin = values["revised"]["adaptive_adwin"]
    revised_ddm = values["revised"]["adaptive_ddm"]

    # Detector quality (alarms / false alarms) now feeds the detector cards.
    detector_summary = [
        {
            "detector": str(row["detector"]).upper(),
            "alarms": int(row["total_alarms"]),
            "falseAlarms": int(row["false_alarms"]),
        }
        for _, row in detector_quality.iterrows()
    ]
    by_detector = {item["detector"]: item for item in detector_summary}

    detector_cards = "\n".join(
        _detector_card_html(
            name,
            by_detector.get(name, {}).get("alarms", "n/a"),
            by_detector.get(name, {}).get("falseAlarms", "n/a"),
        )
        for name in DETECTOR_DETAILS
    )

    initial_table = _comparison_table_html(
        [
            ("Static Random Forest", "static-model", initial_rf, ""),
            ("Adaptive ADWIN", "adwin-model", initial_adwin, ""),
            ("Adaptive DDM", "ddm-model", initial_ddm, ""),
        ]
    )

    revised_table = _comparison_table_html(
        [
            ("Static Random Forest", "static-model", revised_rf, "degraded"),
            ("Adaptive ADWIN", "adwin-model", revised_adwin, "recovered"),
            ("Adaptive DDM", "ddm-model", revised_ddm, "recovered"),
        ]
    )

    # if abs(revised_adwin["drifted"] - revised_ddm["drifted"]) < 5e-4:
    #     adaptive_result = f"reached {revised_adwin['drifted']:.3f}"
    # else:
    #     adaptive_result = (
    #         f"reached {revised_adwin['drifted']:.3f} and "
    #         f"{revised_ddm['drifted']:.3f} respectively"
    #     )

    key_finding = (
        "The original drift did not create meaningful degradation. Under "
        "camouflage drift, the frozen Random Forest degraded "
        "substantially. After detector-triggered retraining, "
        "ADWIN and DDM achieved higher fully-drifted Recall "
        "and F1 than the frozen baseline, showing partial "
        " recovery under concept drift. "
    )

    replacements = {
        "%%KEY_FINDING%%": key_finding,
        "%%ROOT_VARS%%": _root_css_vars(),
        "%%FONT_LINKS%%": GOOGLE_FONT_LINKS,
        "%%DATA_JSON%%": json.dumps(dashboard_data, separators=(",", ":")),
        "%%METRIC_BUTTONS%%": _metric_buttons_html(),
        "%%INITIAL_TABLE%%": initial_table,
        "%%REVISED_TABLE%%": revised_table,
        "%%DETECTOR_CARDS%%": detector_cards,
        "%%INITIAL_RF_TRIO%%": _trio(initial_rf),
        "%%REVISED_RF_TRIO%%": _trio(revised_rf),
        "%%REVISED_RF_RECALL%%": f"{revised_rf['drifted']:.3f}",
        "%%REVISED_ADWIN_RECALL%%": f"{revised_adwin['drifted']:.3f}",
        "%%REVISED_DDM_RECALL%%": f"{revised_ddm['drifted']:.3f}",
        "%%REVISED_RF_F1%%": f"{f1_drifted('static_random_forest'):.3f}",
        "%%REVISED_ADWIN_F1%%": f"{f1_drifted('adaptive_adwin'):.3f}",
        "%%REVISED_DDM_F1%%": f"{f1_drifted('adaptive_ddm'):.3f}",
    }

    html = _DASHBOARD_TEMPLATE

    for token, value in replacements.items():
        html = html.replace(token, value)

    return _write_text(output_path, html)


# ---------------------------------------------------------------------------
# HTML template (plain string: CSS/JS braces need no escaping)
# ---------------------------------------------------------------------------

_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Concept Drift in CNP Fraud Detection</title>
%%FONT_LINKS%%
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
    %%ROOT_VARS%%
}

* { box-sizing: border-box; }

html, body {
    width: 100%;
    min-height: 100%;
    margin: 0;
    overflow-x: hidden;
    overflow-y: auto;
    background: var(--bg);
    color: var(--ink);
    font-family: var(--font);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

button { font: inherit; }

:focus-visible {
    outline: 2px solid var(--initial-ink);
    outline-offset: 2px;
}

.dashboard { width: 100%; min-height: 105vh; padding: 5px; }

.main-page {
    display: grid;
    width: 100%;
    min-height: 105vh;
    grid-template-rows: auto minmax(0, 3fr) minmax(0, 1fr) minmax(0, 5fr);
}

/* ---- Hero ---- */
.hero { padding: 3px 8px 14px; }

.hero h1 {
    margin: 0;
    font-size: clamp(27px, 2.25vw, 40px);
    font-weight: 700;
    line-height: 1.02;
    letter-spacing: -0.005em;
}

.subtitle {
    margin: 3px 0 5px;
    color: var(--accent);
    font-size: clamp(14px, 1.15vw, 20px);
    font-weight: 600;
    letter-spacing: -0.005em;
}

.hero-copy {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 1100px;
    font-size: clamp(10px, 0.78vw, 13px);
    line-height: 1.45;
}

.hero-copy p { margin: 0; }
.hero-copy .definition { color: var(--muted); }

/* ---- Context row ---- */
.context-row {
    display: grid;
    grid-template-columns: 21% 27% 52%;
    gap: 6px;
    min-height: 0;
}

.context-panel {
    min-width: 0;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: auto;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 8px 10px;
}

.context-panel h2 {
    margin: 0 0 6px;
    font-size: clamp(15px, 1.1vw, 19px);
    font-weight: 700;
    letter-spacing: -0.01em;
}

.context-panel p {
    margin: 0 0 6px;
    font-size: clamp(9px, 0.7vw, 11.5px);
    line-height: 1.3;
}

.context-panel strong { color: var(--accent); }

.model-group {
    margin: 0 0 6px;
    font-size: clamp(9px, 0.7vw, 11.5px);
    line-height: 1.3;
}

.model-group:last-child { margin-bottom: 0; }
.model-group b { display: block; margin-bottom: 3px; color: var(--accent); }

.model-item { margin: 0 0 4px 8px; }
.model-name { display: block; font-weight: 600; }
.model-desc { display: block; color: var(--muted); font-size: 0.92em; line-height: 1.25; }

/* ---- Architecture ---- */
.architecture { padding: 6px 8px 3px; }
.architecture h2 { margin: 0 0 2px; }

.architecture-stage {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: calc(100% - 25px);
    overflow: hidden;
}

.architecture svg { display: block; width: 97%; height: 95%; }

.node rect {
    fill: var(--surface);
    stroke: var(--accent);
    stroke-width: 1.15;
    rx: 6;
}

.node text {
    fill: var(--ink);
    font-family: var(--font);
    font-size: 12px;
    text-anchor: middle;
}

.arrow-head { fill: var(--muted); }

.edge {
    fill: none;
    stroke: var(--muted);
    stroke-width: 1.15;
    marker-end: url(#architecture-arrow);
}

.connector { fill: none; stroke: var(--muted); stroke-width: 1.15; }

.edge-label {
    fill: var(--muted);
    font-family: var(--font);
    font-size: 9px;
    text-anchor: middle;
}

.eval-node { cursor: pointer; }

.eval-node rect {
    fill: var(--focus-fill);
    stroke: var(--initial);
    stroke-width: 3;
}

.eval-node text { fill: var(--initial-ink); font-weight: 700; }
.eval-node:hover rect { fill: var(--focus-hover); }

/* ---- Metric row ---- */
.metric-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 10px;
    min-height: 0;
    align-items: center;
    padding: 5px 9px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
}

.metric-controls { min-width: 0; display: grid; grid-template-rows: auto auto auto; gap: 4px; }

.metric-title { margin: 0; font-size: clamp(11px, 0.82vw, 13px); font-weight: 700; }

.metric-buttons { display: flex; align-items: center; flex-wrap: wrap; gap: 5px; }

.metric-button {
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--ink);
    border-radius: 980px;
    padding: 4px 11px;
    cursor: pointer;
    font-size: clamp(8px, 0.64vw, 10px);
    transition: background 0.15s, border-color 0.15s, color 0.15s;
}

.metric-button:hover { border-color: var(--initial); }

.metric-button.active {
    border-color: var(--initial);
    background: var(--initial);
    color: var(--ink);
    font-weight: 600;
}

.metric-explanation {
    margin-top: 1px;
    min-height: 25px;
    padding: 4px 0;
    color: var(--muted);
    font-size: clamp(8px, 0.62vw, 10px);
    line-height: 1.25;
}

.results-button {
    border: 2px solid var(--revised);
    background: var(--revised-tint);
    color: var(--revised);
    border-radius: 980px;
    padding: 8px 15px;
    cursor: pointer;
    font-size: clamp(9px, 0.72vw, 11.5px);
    font-weight: 700;
    white-space: nowrap;
    transition: background 0.15s;
}

.results-button:hover { background: var(--revised-border); }

/* ---- Evaluation area ---- */
.evaluation-area {
    display: grid;
    grid-template-rows: 70% 30%;
    min-height: 0;
    overflow: hidden;
}

.evaluation-top { display: grid; grid-template-columns: 1fr 1fr; min-height: 0; }

.evaluation-column {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    min-width: 0;
    min-height: 0;
    border: 1px solid var(--line);
    padding: 8px 9px;
}

.evaluation-column:first-child { background: var(--initial-tint); border-color: var(--initial-border); }
.evaluation-column:last-child { background: var(--revised-tint); border-color: var(--revised-border); }
.evaluation-column + .evaluation-column { border-left: 2px solid var(--revised-border); }

.evaluation-heading {
    margin: 0 0 2px;
    font-size: clamp(13px, 1vw, 17px);
    font-weight: 700;
    letter-spacing: -0.005em;
}

.evaluation-column:first-child .evaluation-heading { color: var(--initial-ink); }
.evaluation-column:last-child .evaluation-heading { color: var(--revised); }

.evaluation-subtitle { margin: 0 0 5px; color: var(--muted); font-size: clamp(8px, 0.62vw, 10px); }

.chart-grid {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: 7px;
    min-height: 0;
}

.chart-panel {
    min-width: 0;
    min-height: 0;
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 8px;
    overflow: hidden;
}

.evaluation-column:first-child .chart-panel { border-color: var(--initial-border); }
.evaluation-column:last-child .chart-panel { border-color: var(--revised-border); }

.chart-label { margin: 6px 8px 0; font-size: clamp(9px, 0.68vw, 11px); font-weight: 700; }
.chart { width: 100%; height: calc(100% - 21px); }

.explanation-grid { display: grid; grid-template-columns: repeat(4, 1fr); min-height: 0; }

.explanation-box {
    min-width: 0;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: auto;
    padding: 8px 10px;
    border: 1px solid var(--line);
}

.explanation-box:nth-child(-n + 2) { background: var(--initial-tint); border-color: var(--initial-border); }
.explanation-box:nth-child(n + 3) { background: var(--revised-tint); border-color: var(--revised-border); }
.explanation-box + .explanation-box { border-left: 0; }

.explanation-box h3 { margin: 0 0 5px; font-size: clamp(9px, 0.72vw, 12px); font-weight: 700; }
.explanation-box:nth-child(-n + 2) h3 { color: var(--initial-ink); }
.explanation-box:nth-child(n + 3) h3 { color: var(--revised); }

.explanation-box p {
    margin: 0 0 4px;
    color: var(--ink);
    font-size: clamp(7.5px, 0.58vw, 9.5px);
    line-height: 1.35;
}

/* ---- Comparison page ---- */
.comparison-page {
    display: none;
    width: 100%;
    min-height: 100vh;
    padding: 10px 14px 14px;
    grid-template-rows: auto auto auto auto;
    gap: 10px;
}

.comparison-page.active { display: grid; }

.comparison-header { display: flex; align-items: center; gap: 12px; }

.back-button {
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #ffffff;
    padding: 7px 16px;
    cursor: pointer;
    border-radius: 980px;
    font-weight: 600;
    transition: background 0.15s, border-color 0.15s;
}

.back-button:hover { background: #0a626a; border-color: #0a626a; }

.comparison-header h1 {
    margin: 0;
    font-size: clamp(21px, 1.9vw, 31px);
    font-weight: 700;
    letter-spacing: -0.01em;
}

.comparison-evaluations { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

.comparison-evaluation {
    border: 1px solid var(--initial-border);
    background: var(--initial-tint);
    border-radius: 12px;
    padding: 13px 15px;
}

.comparison-evaluation.revised { background: var(--revised-tint); border-color: var(--revised-border); }

.comparison-evaluation h2 {
    margin: 0 0 4px;
    font-size: clamp(17px, 1.35vw, 23px);
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--initial-ink);
}

.comparison-evaluation.revised h2 { color: var(--revised); }

.comparison-intro {
    margin: 0 0 12px;
    color: var(--muted);
    font-size: clamp(10px, 0.78vw, 13px);
    line-height: 1.4;
}

.table-metric {
    margin: 0 0 5px;
    font-size: clamp(11px, 0.85vw, 14px);
    font-weight: 700;
}

.model-comparison-table {
    display: grid;
    grid-template-columns: minmax(150px, 1.35fr) repeat(3, minmax(80px, 1fr));
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 8px;
    overflow: hidden;
}

.comparison-cell {
    min-width: 0;
    padding: 9px 8px;
    border-right: 1px solid var(--line-soft);
    border-bottom: 1px solid var(--line-soft);
    text-align: center;
    font-size: clamp(10px, 0.78vw, 13px);
}

.comparison-cell:nth-child(4n) { border-right: 0; }
.comparison-cell.header { background: var(--bg); color: var(--muted); font-weight: 600; }
.comparison-cell.model { text-align: left; font-weight: 600; }
.comparison-cell.static-model { border-left: 4px solid var(--static); }
.comparison-cell.adwin-model { border-left: 4px solid var(--initial); }
.comparison-cell.ddm-model { border-left: 4px solid var(--revised); }

.comparison-cell.value {
    font-size: clamp(14px, 1.1vw, 18px);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

.comparison-cell.degraded { color: var(--bad); }
.comparison-cell.recovered { color: var(--good); }

.comparison-interpretation {
    margin: 12px 0 0;
    padding: 0;
    font-size: clamp(10px, 0.75vw, 12px);
    line-height: 1.5;
}

.recovery-section {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 13px 15px;
}

.recovery-section h2 {
    margin: 0 0 5px;
    font-size: clamp(17px, 1.35vw, 22px);
    font-weight: 700;
    letter-spacing: -0.01em;
}

.recovery-subtitle { margin: 0 0 10px; color: var(--muted); font-size: clamp(10px, 0.78vw, 13px); }

.recovery-grid { display: grid; grid-template-columns: 1fr 1fr 1.25fr; gap: 10px; }

.detector-card,
.recovery-result {
    border: 1px solid var(--line-soft);
    border-radius: 8px;
    padding: 10px 12px;
    background: var(--bg);
}

.detector-card:nth-child(1) { border-top: 3px solid var(--initial); }
.detector-card:nth-child(2) { border-top: 3px solid var(--revised); }
.recovery-result {
    border-color: var(--recovery-border);
    border-top: 3px solid var(--recovery);
    background: var(--recovery-tint);
}

.detector-card h3,
.recovery-result h3 { margin: 0 0 7px; font-size: clamp(13px, 0.95vw, 16px); font-weight: 700; }
.recovery-result h3 { color: var(--recovery); }

.detector-card p,
.recovery-result p { margin: 3px 0; font-size: clamp(9px, 0.72vw, 12px); line-height: 1.35; }

.key-finding {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 6px solid var(--accent);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: clamp(12px, 0.95vw, 15px);
    line-height: 1.55;
}

.key-finding strong { color: var(--accent); }

.recovery-arrow { margin: 8px 0; font-size: clamp(11px, 0.8vw, 13px); font-weight: 700; color: var(--recovery); }

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
</style>
</head>

<body>
<div class="dashboard">

<main class="main-page" id="main-page">

<section class="hero">
<h1>Concept Drift in CNP Fraud Detection</h1>
<div class="subtitle">Static vs. adaptive model performance under controlled concept-drift scenarios</div>
<div class="hero-copy">
<p><strong>Purpose:</strong> Compare how static and adaptive fraud models behave across controlled concept-drift phases and determine whether adaptation helps as transaction behavior changes.</p>
<p class="definition"><strong>Concept drift:</strong> the relationship between transaction characteristics and fraud changes over time, so models trained on earlier behavior may become less effective.</p>
</div>
</section>

<section class="context-row">

<div class="context-panel">
<h2>Dataset</h2>
<p><strong>CreditTransAct</strong><br>Source transaction dataset used by the project.</p>
<p><strong>CNP transaction stream</strong><br>Card-Not-Present transactions used for streaming fraud detection.</p>
<p><strong>500,000 warm-up rows</strong><br>Historical rows used before stream evaluation.</p>
<p><strong>1.5M evaluation rows</strong><br>Transactions evaluated across the controlled-drift stream.</p>
<p><strong>50,000 rows per batch</strong><br>Sequential evaluation batch size.</p>
<p><strong>30 evaluation batches</strong><br>Evaluation stream processed across 30 sequential batches.</p>
<p><strong>7-batch label delay</strong><br>Fraud labels become available seven batches after prediction.</p>
</div>

<div class="context-panel">
<h2>Models Compared</h2>
<div class="model-group"><b>1. Frozen Static Models</b>
<div class="model-item"><span class="model-name">Logistic Regression</span><span class="model-desc">Simple linear baseline; frozen after warm-up.</span></div>
<div class="model-item"><span class="model-name">Random Forest</span><span class="model-desc">Nonlinear baseline; frozen after warm-up.</span></div>
</div>
<div class="model-group"><b>2. Adaptive Models</b>
<div class="model-item"><span class="model-name">Adaptive None</span><span class="model-desc">Online updates, but no drift detection.</span></div>
<div class="model-item"><span class="model-name">Adaptive ADWIN</span><span class="model-desc">ADWIN detection + detector-triggered retraining.</span></div>
<div class="model-item"><span class="model-name">Adaptive DDM</span><span class="model-desc">DDM detection + detector-triggered retraining.</span></div>
</div>
<div class="model-group"><b>3. Online Ensemble</b>
<div class="model-item"><span class="model-name">5× Incremental Logistic Regression</span><span class="model-desc">Online bagging; continuously updates, no drift detector.</span></div>
</div>
</div>

<div class="context-panel architecture">
<h2>System Architecture</h2>
<div class="architecture-stage">
<svg viewBox="0 0 1280 360" role="img" aria-label="Streaming fraud detection system architecture">
<defs>
<marker id="architecture-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
<path class="arrow-head" d="M0,0 L0,6 L8,3 z"/>
</marker>
</defs>

<g class="node"><rect x="10" y="125" width="120" height="75"/>
<text x="70" y="150">Seeded</text><text x="70" y="168">sequential</text><text x="70" y="186">stream</text></g>

<path class="edge" d="M130 162 L150 162 L150 68 L175 68"/>
<path class="edge" d="M130 162 L150 162 L150 280 L175 280"/>

<g class="node"><rect x="175" y="38" width="135" height="60"/>
<text x="242" y="63">500,000-row</text><text x="242" y="81">warm-up</text></g>

<path class="edge" d="M310 68 L340 68"/>

<g class="node"><rect x="340" y="38" width="135" height="60"/>
<text x="407" y="63">Frozen</text><text x="407" y="81">preprocessing</text></g>

<path class="connector" d="M475 68 L1035 68"/>
<path class="connector" d="M1035 22 L1035 160"/>

<path class="edge" d="M1035 22 L1060 22"/>
<g class="node"><rect x="1060" y="4" width="125" height="36"/>
<text x="1122" y="27">Static Logistic</text></g>

<path class="edge" d="M1035 68 L1060 68"/>
<g class="node"><rect x="1060" y="46" width="125" height="44"/>
<text x="1122" y="65">Static Random</text><text x="1122" y="81">Forest</text></g>

<path class="edge" d="M1035 114 L1060 114"/>
<g class="node"><rect x="1060" y="96" width="125" height="36"/>
<text x="1122" y="119">Online Ensemble</text></g>

<path class="edge" d="M1035 160 L1060 160"/>
<g class="node"><rect x="1060" y="138" width="125" height="44"/>
<text x="1122" y="157">Adaptive</text><text x="1122" y="173">models</text></g>

<g class="node"><rect x="175" y="247" width="135" height="66"/>
<text x="242" y="271">50,000-row</text><text x="242" y="289">evaluation batches</text></g>

<path class="edge" d="M310 280 L340 280"/>

<g class="node"><rect x="340" y="247" width="135" height="66"/>
<text x="407" y="272">Controlled drift</text><text x="407" y="290">scenario</text></g>

<path class="edge" d="M475 280 L520 280"/>

<g class="node"><rect x="520" y="247" width="120" height="66"/>
<text x="580" y="272">Model</text><text x="580" y="290">predictions</text></g>

<path class="edge" d="M640 280 L665 280 L665 220 L690 220"/>

<g class="node"><rect x="690" y="190" width="125" height="60"/>
<text x="752" y="214">Seven-batch</text><text x="752" y="232">label delay</text></g>

<path class="edge" d="M815 220 L840 220"/>

<g class="node"><rect x="840" y="190" width="125" height="60"/>
<text x="902" y="225">ADWIN and DDM</text></g>

<path class="edge" d="M965 220 L990 220"/>
<text class="edge-label" x="977" y="183">Confirmed alarm</text>

<g class="node"><rect x="990" y="190" width="135" height="60"/>
<text x="1057" y="214">Recent-window</text><text x="1057" y="232">retraining</text></g>

<path class="edge" d="M1125 220 L1215 220 L1215 160 L1185 160"/>
<text class="edge-label" x="1170" y="212">Retrained</text>

<path class="edge" d="M640 280 L665 280 L665 330 L690 330"/>

<g class="node eval-node" id="evaluation-node" tabindex="0" role="button" aria-label="Focus evaluation results">
<title>Focus the evaluation results below</title>
<rect x="690" y="302" width="125" height="55"/>
<text x="752" y="324">EVALUATION</text><text x="752" y="343">Explore results →</text></g>

</svg>
</div>
</div>

</section>

<section class="metric-row" id="metric-row">
<div class="metric-controls">
<div class="metric-title">Evaluation Metrics</div>
<div class="metric-buttons">
%%METRIC_BUTTONS%%
</div>
<div class="metric-explanation" id="initial-metric-explanation"></div>
</div>
<button class="results-button" id="results-button" type="button">RESULTS BETWEEN THE TWO EVALUATIONS →</button>
</section>

<section class="evaluation-area" id="evaluation-area">

<div class="evaluation-top">

<div class="evaluation-column">
<div>
<h2 class="evaluation-heading">Initial Evaluation — Original Controlled Drift</h2>
<p class="evaluation-subtitle">Performance across pre-drift, ramp-drift, and fully drifted phases.</p>
</div>
<div class="chart-grid">
<div class="chart-panel"><div class="chart-label">Model Performance Across Drift Phases</div><div class="chart" id="initial-performance"></div></div>
<div class="chart-panel"><div class="chart-label">Relative Loss</div><div class="chart" id="initial-loss"></div></div>
</div>
</div>

<div class="evaluation-column">
<div>
<h2 class="evaluation-heading">Revised Evaluation — Camouflage Drift</h2>
<p class="evaluation-subtitle">Performance across pre-drift, ramp-drift, and fully drifted phases.</p>
</div>
<div class="chart-grid">
<div class="chart-panel"><div class="chart-label">Model Performance Across Drift Phases</div><div class="chart" id="revised-performance"></div></div>
<div class="chart-panel"><div class="chart-label">Relative Loss</div><div class="chart" id="revised-loss"></div></div>
</div>
</div>

</div>

<div class="explanation-grid">

<div class="explanation-box">
<h3>Why was Static Random Forest already extremely strong?</h3>
<p>The initial evaluation showed near-perfect pre-drift performance because the data contained highly discriminative fraud-related attributes.</p>
<p>Follow-up feature audit and ablation work found no evidence that this strength came from direct target leakage.</p>
</div>

<div class="explanation-box">
<h3>Why did Static Random Forest improve after the original drift?</h3>
<p>The original drift increased fraud-versus-legitimate separation across several fraud-related features, making fraud easier rather than harder to distinguish.</p>
<p>Static RF Recall: <strong>%%INITIAL_RF_TRIO%%</strong></p>
<p>Other models also remained near their pre-drift performance, leaving limited degradation for adaptive models to recover from.</p>
</div>

<div class="explanation-box">
<h3>Why camouflage?</h3>
<p>The original drift made fraud easier to distinguish. Camouflage drift instead moves fraudulent behavior closer to legitimate behavior, creating meaningful performance degradation.</p>
<p>This allows us to evaluate whether drift detection and adaptive retraining can recover lost performance.</p>
</div>

<div class="explanation-box">
<h3>What changed under camouflage drift?</h3>
<p>Camouflage moves selected fraudulent feature distributions toward legitimate behavior, reducing class separation and creating meaningful performance degradation.</p>
<p>Unlike the original drift, the revised scenario produces positive relative performance loss. The frozen models experience greater degradation, while ADWIN and DDM use drift detection and adaptive retraining to partially recover performance.</p>
</div>

</div>

</section>

</main>

<section class="comparison-page" id="comparison-page">

<div class="comparison-header">
<button class="back-button" id="comparison-back" type="button">← Dashboard</button>
<h1>Results Between the Two Evaluations</h1>
</div>

<div class="key-finding">
<strong>Key Finding:</strong> %%KEY_FINDING%%
</div>

<div class="comparison-evaluations">

<div class="comparison-evaluation">
<h2>Initial Evaluation — Original Controlled Drift</h2>
<p class="comparison-intro">The original controlled drift did not create meaningful degradation. Static Random Forest, Adaptive ADWIN, and Adaptive DDM all remained extremely strong across the drift phases.</p>
%%INITIAL_TABLE%%
<p class="comparison-interpretation"><strong>Interpretation:</strong> the original drift increased class separability instead of making fraud harder to detect. Because there was no meaningful degradation, the first evaluation could not clearly demonstrate the benefit of detector-triggered adaptation.</p>
</div>

<div class="comparison-evaluation revised">
<h2>Revised Evaluation — Camouflage Drift</h2>
<p class="comparison-intro">Camouflage drift created the intended performance pressure. The frozen Random Forest degraded, while ADWIN- and DDM-driven adaptive models recovered more fraud in the fully drifted phase.</p>
%%REVISED_TABLE%%
<p class="comparison-interpretation"><strong>Interpretation:</strong> during the ramp phase the frozen Random Forest still had higher Recall, but after detector-triggered retraining the ADWIN and DDM adaptive models reached fully drifted Recall <strong>%%REVISED_ADWIN_RECALL%%</strong>, compared with <strong>%%REVISED_RF_RECALL%%</strong> for the frozen Random Forest.</p>
</div>

</div>

<div class="recovery-section">
<h2>Concept-Drift Detection and Recovery Evidence</h2>
<p class="recovery-subtitle">The revised evaluation shows what happened after the detectors observed the degradation through delayed labels.</p>

<div class="recovery-grid">

%%DETECTOR_CARDS%%

<div class="recovery-result">
<h3>Recovery Result</h3>
<p><strong>Static Random Forest</strong><br>Fully drifted Recall: %%REVISED_RF_RECALL%%<br>Fully drifted F1: %%REVISED_RF_F1%%</p>
<div class="recovery-arrow">Drift detected → recent-window retraining → partial recovery</div>
<p><strong>Adaptive ADWIN</strong><br>Fully drifted Recall: %%REVISED_ADWIN_RECALL%%<br>Fully drifted F1: %%REVISED_ADWIN_F1%%</p>
<p><strong>Adaptive DDM</strong><br>Fully drifted Recall: %%REVISED_DDM_RECALL%%<br>Fully drifted F1: %%REVISED_DDM_F1%%</p>
<p><strong>Conclusion:</strong> detector-triggered adaptation recovered part of the fraud-detection performance lost under camouflage drift. In this run, ADWIN and DDM produced the same recovery result.</p>
</div>

</div>
</div>

</section>

</div>

<script>
(() => {
    const root = document.querySelector(".dashboard");

    if (!root || root.dataset.initialized === "true") {
        return;
    }

    root.dataset.initialized = "true";

    const data = %%DATA_JSON%%;
    const { palette, phaseColors, font } = data;
    const $ = (selector) => root.querySelector(selector);

    const mainPage = $("#main-page");
    const comparisonPage = $("#comparison-page");
    const metricButtons = Array.from(root.querySelectorAll(".metric-button"));
    const metricExplanation = $("#initial-metric-explanation");
    const evaluationArea = $("#evaluation-area");
    const evaluationNode = $("#evaluation-node");

    const chartConfig = { responsive: true, displaylogo: false, displayModeBar: false };
    const tickFont = (size) => ({ family: font, size });
    const hoverLabel = {
        font: { family: font, size: 9, color: "#ffffff" },
        align: "left",
        bordercolor: palette.surface,
    };
    const decimals = (metric) => (metric === "fpr" ? 4 : 3);

    const performanceLayout = (metric) => ({
        barmode: "group",
        autosize: true,
        margin: { l: 46, r: 12, t: 58, b: 78 },
        xaxis: { tickangle: -18, automargin: true, tickfont: tickFont(10), fixedrange: true },
        yaxis: {
            title: { text: data.metricLabels[metric], font: tickFont(9), standoff: 4 },
            range: metric === "fpr" ? undefined : [0, 1.12],
            tickformat: metric === "fpr" ? ".4f" : ".1f",
            gridcolor: palette.line_soft,
            automargin: true,
            tickfont: tickFont(8),
            fixedrange: true,
        },
        legend: { orientation: "h", x: 0, xanchor: "left", y: 1.08, yanchor: "bottom", font: tickFont(9) },
        font: { family: font, size: 10, color: palette.ink },
        hoverlabel: hoverLabel,
        bargap: 0.2,
        bargroupgap: 0.04,
        paper_bgcolor: palette.surface,
        plot_bgcolor: palette.plot_bg,
    });

    const lossLayout = (metric, values) => ({
        autosize: true,
        margin: { l: 48, r: 12, t: 22, b: 84 },
        xaxis: { tickangle: -28, automargin: true, tickfont: tickFont(9), fixedrange: true },
        yaxis: {
            title: {
                text: metric === "fpr" ? "Relative FPR Increase (%)" : "Relative Loss (%)",
                font: tickFont(9),
                standoff: 4,
            },
            range: [Math.min(-5, ...values) - 5, Math.max(5, ...values) + 7],
            gridcolor: palette.line_soft,
            automargin: true,
            tickfont: tickFont(8),
            fixedrange: true,
        },
        font: { family: font, size: 9, color: palette.ink },
        hoverlabel: hoverLabel,
        bargap: 0.28,
        paper_bgcolor: palette.surface,
        plot_bgcolor: palette.plot_bg,
        showlegend: false,
        shapes: [{
            type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0,
            line: { color: palette.muted, dash: "dash", width: 1 },
        }],
    });

    const performanceTraces = (scenario, metric) => {
        const values = data[scenario][metric].performance;
        const digits = decimals(metric);
        const colors = phaseColors;

        return [
            ["Pre-drift", "pre"],
            ["Ramp drift", "ramp"],
            ["Fully drifted", "drifted"],
        ].map(([name, key], index) => ({
            type: "bar",
            name,
            x: values.labels,
            y: values[key],
            marker: { color: colors[index] },
            text: values[key].map((value) => value.toFixed(digits)),
            textposition: "outside",
            cliponaxis: false,
            textfont: tickFont(9),
            hovertemplate: "%{x}<br>" + name + ": %{y:.4f}<extra></extra>",
        }));
    };

    // Positive = worse (red); negative = improvement (green).
    const lossTrace = (scenario, metric) => {
        const values = data[scenario][metric].loss;

        return [{
            type: "bar",
            x: values.labels,
            y: values.values,
            marker: { color: values.values.map((v) => (v > 0 ? palette.bad : palette.good)) },
            text: values.values.map((value) => value.toFixed(1) + "%"),
            textposition: "outside",
            cliponaxis: false,
            textfont: tickFont(9),
            hovertemplate: "%{x}<br>%{y:.2f}%<extra></extra>",
        }];
    };

    const renderMetric = (metric) => {
        metricButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.metric === metric);
        });

        for (const scenario of ["initial", "revised"]) {
            Plotly.react(
                $("#" + scenario + "-performance"),
                performanceTraces(scenario, metric),
                performanceLayout(metric),
                chartConfig
            );

            Plotly.react(
                $("#" + scenario + "-loss"),
                lossTrace(scenario, metric),
                lossLayout(metric, data[scenario][metric].loss.values),
                chartConfig
            );
        }

        metricExplanation.innerHTML =
            "<strong>" + data.metricLabels[metric] + ":</strong> " + data.metricExplanations[metric];
    };

    metricButtons.forEach((button) => {
        button.addEventListener("click", () => renderMetric(button.dataset.metric));
    });

    const focusEvaluation = () => {
        evaluationArea.animate(
            [
                { outline: "3px solid " + palette.initial },
                { outline: "3px solid transparent" },
            ],
            { duration: 900, easing: "ease-out" }
        );
    };

    evaluationNode.addEventListener("click", focusEvaluation);

    evaluationNode.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            focusEvaluation();
        }
    });

    const showPage = (showComparison) => {
        comparisonPage.classList.toggle("active", showComparison);
        mainPage.style.display = showComparison ? "none" : "grid";
        window.scrollTo(0, 0);
        window.dispatchEvent(new Event("resize"));
    };

    $("#results-button").addEventListener("click", () => showPage(true));
    $("#comparison-back").addEventListener("click", () => showPage(false));

    let currentMetric = "recall";
    metricButtons.forEach((button) => {
        button.addEventListener("click", () => { currentMetric = button.dataset.metric; });
    });

    renderMetric(currentMetric);

    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(() => renderMetric(currentMetric));
    }
})();
</script>

</body>
</html>
"""
