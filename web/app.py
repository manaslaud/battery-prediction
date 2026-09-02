"""Flask portal: landing page plus hybrid battery SOH / remaining-life predictor."""

import math
import sys
from pathlib import Path

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battery_soh_predictor import load_model, predict_life, predict_soh  # noqa: E402

TEMP_MIN_C = -20.0
TEMP_MAX_C = 80.0

app = Flask(__name__)

_bundle = None
_load_error = None
try:
    _bundle = load_model()
except FileNotFoundError as exc:
    _load_error = str(exc)


def _soh_level(soh):
    if soh >= 0.90:
        return "healthy"
    if soh >= 0.80:
        return "caution"
    return "low"


def _defaults():
    if _bundle is None:
        return {"max_cycle": "", "max_throughput": ""}
    _, _, ref_max_cycle, ref_max_throughput = _bundle
    return {
        "max_cycle": f"{ref_max_cycle:.0f}",
        "max_throughput": f"{ref_max_throughput:.1f}",
    }


def _parse_required_float(raw, label):
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"{label} is required.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return value


def _parse_form(form):
    cycle = _parse_required_float(form.get("cycle"), "Cycle number")
    throughput = _parse_required_float(form.get("throughput"), "Cumulative Ah throughput")
    avg_temp = _parse_required_float(form.get("avg_temp"), "Average temperature")
    max_cycle = _parse_required_float(form.get("max_cycle"), "Expected end-of-life cycle count")
    max_throughput = _parse_required_float(
        form.get("max_throughput"), "Expected end-of-life Ah throughput"
    )

    for label, value in (
        ("Cycle number", cycle),
        ("Cumulative Ah throughput", throughput),
        ("Expected end-of-life cycle count", max_cycle),
        ("Expected end-of-life Ah throughput", max_throughput),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be greater than 0.")

    if not (TEMP_MIN_C <= avg_temp <= TEMP_MAX_C):
        raise ValueError(
            f"Average temperature must be between {TEMP_MIN_C:.0f} and {TEMP_MAX_C:.0f} °C."
        )

    estimate_life = form.get("estimate_life") == "on"
    custom_raw = (form.get("custom_threshold") or "").strip()
    custom_threshold = None
    if custom_raw:
        custom_threshold = _parse_required_float(custom_raw, "Custom SOH threshold")
        if not (0 < custom_threshold < 1):
            raise ValueError("Custom SOH threshold must be between 0 and 1 (for example 0.7).")

    return {
        "cycle": cycle,
        "throughput": throughput,
        "avg_temp": avg_temp,
        "max_cycle": max_cycle,
        "max_throughput": max_throughput,
        "estimate_life": estimate_life,
        "custom_threshold": custom_threshold,
    }


def _form_values(parsed=None, form=None):
    values = {
        "cycle": "",
        "throughput": "",
        "avg_temp": "",
        "estimate_life": True,
        "custom_threshold": "",
        **_defaults(),
    }
    if parsed is not None:
        values.update(
            cycle=parsed["cycle"],
            throughput=parsed["throughput"],
            avg_temp=parsed["avg_temp"],
            max_cycle=parsed["max_cycle"],
            max_throughput=parsed["max_throughput"],
            estimate_life=parsed["estimate_life"],
            custom_threshold="" if parsed["custom_threshold"] is None else parsed["custom_threshold"],
        )
    elif form is not None:
        for key in ("cycle", "throughput", "avg_temp", "max_cycle", "max_throughput", "custom_threshold"):
            if form.get(key) is not None:
                values[key] = form.get(key)
        values["estimate_life"] = form.get("estimate_life") == "on"
    return values


def _render_predict(form, error=None, soh=None, life=None, status=200):
    soh_level = _soh_level(soh["hybrid_soh"]) if soh else None
    physics_level = _soh_level(soh["physics_soh"]) if soh else None
    return render_template(
        "predict.html",
        active="predict",
        load_error=_load_error,
        error=error,
        form=form,
        soh=soh,
        soh_level=soh_level,
        physics_level=physics_level,
        life=life,
    ), status


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html", active="home")


@app.route("/predict", methods=["GET"])
def predict_get():
    return _render_predict(_form_values())


@app.route("/predict", methods=["POST"])
def predict_post():
    if _bundle is None:
        return _render_predict(_form_values(form=request.form), status=503)

    model, scaler, _, _ = _bundle
    try:
        parsed = _parse_form(request.form)
    except ValueError as exc:
        return _render_predict(_form_values(form=request.form), error=str(exc), status=400)

    soh = predict_soh(
        model, scaler,
        parsed["cycle"], parsed["throughput"], parsed["avg_temp"],
        parsed["max_cycle"], parsed["max_throughput"],
    )
    life = None
    if parsed["estimate_life"]:
        thresholds = [0.80, 0.50]
        custom = parsed["custom_threshold"]
        if custom is not None and custom not in thresholds:
            thresholds.append(custom)
        life = predict_life(
            model, scaler,
            parsed["cycle"], parsed["throughput"], parsed["avg_temp"],
            parsed["max_cycle"], parsed["max_throughput"],
            soh["hybrid_soh"], thresholds,
        )

    return _render_predict(_form_values(parsed=parsed), soh=soh, life=life)


if __name__ == "__main__":
    app.run(debug=True)
