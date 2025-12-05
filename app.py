from flask import Flask, render_template, request, jsonify, send_file, url_for
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from io import BytesIO
import datetime

# =========================
# 1. 路径与资源
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_knn.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# =========================
# 2. 加载模型
# =========================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# 3. 配置
# =========================
FEATURE_KEYS = ["体斜长", "胸围", "腹围", "腰角宽", "体高"]

# 训练数据范围（可后续替换为训练集 P1~P99）
FEATURE_MIN = {"体斜长": 114.4, "胸围": 142.0, "腹围": 153.0, "腰角宽": 31.0, "体高": 103.1}
FEATURE_MAX = {"体斜长": 173.0, "胸围": 216.0, "腹围": 248.0, "腰角宽": 61.0, "体高": 141.0}

app = Flask(__name__)

# =========================
# 4. 工具函数
# =========================
def get_type_label(score_int: int) -> str:
    if score_int <= 3:
        return "偏瘦型"
    elif 4 <= score_int <= 6:
        return "理想型"
    elif score_int == 7:
        return "偏肥型"
    else:
        return "肥胖型"

def parse_inputs(data_dict):
    """
    data_dict: dict-like (来自 form 或 json)
    return: (inputs, errors, warnings)
    """
    inputs = {}
    errors = []
    warnings = []

    for k in FEATURE_KEYS:
        raw = data_dict.get(k, None)
        if raw is None or str(raw).strip() == "":
            errors.append(f"{k} 不能为空")
            continue
        try:
            val = float(raw)
        except Exception:
            errors.append(f"{k} 必须为数字")
            continue
        if val <= 0:
            errors.append(f"{k} 必须为正数")
            continue
        inputs[k] = val

    if errors:
        return None, errors, warnings

    # 合理性/范围提示（不直接粗暴判死）
    for k in FEATURE_KEYS:
        v = inputs[k]
        if v < FEATURE_MIN[k] or v > FEATURE_MAX[k]:
            warnings.append(
                f"{k} 超出训练数据范围（{FEATURE_MIN[k]} ~ {FEATURE_MAX[k]} cm），结果需谨慎解读"
            )

    if inputs["体高"] < 50:
        warnings.append("体高数值偏小，请确认单位为 cm")

    return inputs, errors, warnings

def build_features(inputs):
    # 派生特征：胸围/体高
    chest_height = inputs["胸围"] / inputs["体高"]
    X = np.array(
        [[inputs["体斜长"], inputs["胸围"], inputs["腹围"], inputs["腰角宽"], chest_height]],
        dtype=float
    )
    return X

def predict_from_inputs(inputs):
    X = build_features(inputs)
    Xs = scaler.transform(X)

    pred = model.predict(Xs)[0]
    score_int = int(round(float(pred)))
    score_int = max(1, min(9, score_int))

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(Xs)[0]
            confidence = float(np.max(proba))
        except Exception:
            confidence = None

    tol_min = max(1, score_int - 1)
    tol_max = min(9, score_int + 1)

    return score_int, confidence, tol_min, tol_max

def confidence_level(conf):
    if conf is None:
        return ""
    if conf >= 0.75:
        return "高"
    if conf >= 0.55:
        return "中"
    return "低"

# =========================
# 5. 单牛页面
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    type_label = None
    tolerance_hint = ""
    error = ""
    warnings = []
    confidence_text = ""
    inputs = {}

    if request.method == "POST":
        inputs, errors, warnings = parse_inputs(request.form)
        if errors:
            error = "；".join(errors)
            return render_template(
                "index.html",
                score=score,
                type_label=type_label,
                tolerance_hint=tolerance_hint,
                error=error,
                warnings=warnings,
                confidence_text=confidence_text,
                inputs=request.form
            )

        try:
            score_int, conf, tol_min, tol_max = predict_from_inputs(inputs)
            score = str(score_int)
            type_label = get_type_label(score_int)

            # 生产口径提示（你论文的 4–8 核心区间）
            tolerance_hint = f"±1容错范围：{max(4, tol_min)}~{min(8, tol_max)}分"

            if conf is not None:
                confidence_text = f"预测置信度：{confidence_level(conf)}（{conf:.2f}）"

        except Exception as e:
            error = f"计算失败，请检查输入（{e}）"

    return render_template(
        "index.html",
        score=score,
        type_label=type_label,
        tolerance_hint=tolerance_hint,
        error=error,
        warnings=warnings,
        confidence_text=confidence_text,
        inputs=inputs
    )

# =========================
# 6. JSON API
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    inputs, errors, warnings = parse_inputs(data or {})

    if errors:
        return jsonify({
            "ok": False,
            "errors": errors,
            "warnings": warnings
        }), 400

    try:
        score_int, conf, tol_min, tol_max = predict_from_inputs(inputs)
        return jsonify({
            "ok": True,
            "inputs": inputs,
            "score": score_int,
            "type_label": get_type_label(score_int),
            "tolerance": {"min": tol_min, "max": tol_max},
            "tolerance_prod": {"min": max(4, tol_min), "max": min(8, tol_max)},
            "confidence": conf,
            "confidence_level": confidence_level(conf) if conf is not None else None,
            "warnings": warnings
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "errors": [str(e)],
            "warnings": warnings
        }), 500

# =========================
# 7. 批量页面
# =========================
@app.route("/batch", methods=["GET"])
def batch_page():
    return render_template("batch.html")

@app.route("/batch/predict", methods=["POST"])
def batch_predict():
    if "file" not in request.files:
        return render_template("batch.html", error="未检测到文件，请重新上传")

    f = request.files["file"]
    if f.filename == "":
        return render_template("batch.html", error="文件名为空，请重新选择文件")

    try:
        df = pd.read_csv(f)
    except Exception:
        # 尝试 Excel
        try:
            f.stream.seek(0)
            df = pd.read_excel(f)
        except Exception as e:
            return render_template("batch.html", error=f"文件解析失败：{e}")

    missing = [c for c in FEATURE_KEYS if c not in df.columns]
    if missing:
        return render_template(
            "batch.html",
            error=f"缺少必要列：{', '.join(missing)}。请确保表头为：{', '.join(FEATURE_KEYS)}"
        )

    scores, types, tol_mins, tol_maxs, confs, notes = [], [], [], [], [], []

    for _, row in df.iterrows():
        row_dict = {k: row.get(k, None) for k in FEATURE_KEYS}
        inputs, errors, warnings = parse_inputs(row_dict)

        if errors:
            scores.append(np.nan)
            types.append("数据错误")
            tol_mins.append(np.nan)
            tol_maxs.append(np.nan)
            confs.append(np.nan)
            notes.append("；".join(errors))
            continue

        try:
            score_int, conf, tol_min, tol_max = predict_from_inputs(inputs)
            scores.append(score_int)
            types.append(get_type_label(score_int))
            tol_mins.append(tol_min)
            tol_maxs.append(tol_max)
            confs.append(conf if conf is not None else np.nan)
            notes.append("；".join(warnings) if warnings else "")
        except Exception as e:
            scores.append(np.nan)
            types.append("预测失败")
            tol_mins.append(np.nan)
            tol_maxs.append(np.nan)
            confs.append(np.nan)
            notes.append(str(e))

    out = df.copy()
    out["预测BCS"] = scores
    out["评分类型"] = types
    out["容错下限"] = tol_mins
    out["容错上限"] = tol_maxs
    out["置信度"] = confs
    out["备注"] = notes

    bio = BytesIO()
    out.to_csv(bio, index=False, encoding="utf-8-sig")
    bio.seek(0)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"BCS_batch_result_{stamp}.csv"

    return send_file(
        bio,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )

# =========================
# 8. 单牛 PDF 简报
# =========================
def build_pdf_report(inputs, score_int, type_label, tol_min, tol_max, conf, warnings):
    """
    使用 reportlab 生成简报。
    为避免中文字体问题：
    - 若你提供 static/fonts/SimHei.ttf 或 msyh.ttc，可自动注册
    - 否则可能出现中文乱码/方框
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception as e:
        raise RuntimeError("未安装 reportlab，请执行：pip install reportlab") from e

    font_candidates = [
        BASE_DIR / "static" / "fonts" / "SimHei.ttf",
        BASE_DIR / "static" / "fonts" / "msyh.ttc",
        BASE_DIR / "static" / "fonts" / "MicrosoftYaHei.ttf",
    ]
    font_name = "Helvetica"
    for fp in font_candidates:
        if fp.exists():
            try:
                pdfmetrics.registerFont(TTFont("CNFont", str(fp)))
                font_name = "CNFont"
                break
            except Exception:
                pass

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont(font_name, 16)
    c.drawString(2.2*cm, height - 2.2*cm, "肉牛体况智能评价 - 单牛简报")

    c.setFont(font_name, 10)
    c.drawString(2.2*cm, height - 3.0*cm, f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 4.2*cm
    c.setFont(font_name, 12)
    c.drawString(2.2*cm, y, "输入体尺（cm）")
    y -= 0.7*cm

    c.setFont(font_name, 11)
    for k in FEATURE_KEYS:
        c.drawString(2.6*cm, y, f"{k}：{inputs[k]:.1f}")
        y -= 0.55*cm

    y -= 0.2*cm
    c.setFont(font_name, 12)
    c.drawString(2.2*cm, y, "智能预测结果")
    y -= 0.75*cm

    c.setFont(font_name, 11)
    c.drawString(2.6*cm, y, f"预测BCS：{score_int}")
    y -= 0.55*cm
    c.drawString(2.6*cm, y, f"评分类型：{type_label}")
    y -= 0.55*cm
    c.drawString(2.6*cm, y, f"±1容错范围：{tol_min}~{tol_max}")
    y -= 0.55*cm
    c.drawString(2.6*cm, y, f"生产口径容错：{max(4, tol_min)}~{min(8, tol_max)}")
    y -= 0.55*cm

    if conf is not None:
        c.drawString(2.6*cm, y, f"预测置信度：{confidence_level(conf)}（{conf:.2f}）")
        y -= 0.55*cm

    if warnings:
        y -= 0.1*cm
        c.setFont(font_name, 11)
        c.drawString(2.2*cm, y, "提示：")
        y -= 0.55*cm
        c.setFont(font_name, 10)
        for w in warnings[:6]:
            c.drawString(2.6*cm, y, f"- {w}")
            y -= 0.48*cm

    c.setFont(font_name, 9)
    c.drawString(2.2*cm, 1.8*cm, "本简报为科研原型输出，建议结合现场经验与生产管理策略综合判断。")

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer

@app.route("/report", methods=["POST"])
def single_report():
    inputs, errors, warnings = parse_inputs(request.form)
    if errors:
        return render_template(
            "index.html",
            score=None,
            type_label=None,
            tolerance_hint="",
            error="；".join(errors),
            warnings=warnings,
            confidence_text="",
            inputs=request.form
        )

    try:
        score_int, conf, tol_min, tol_max = predict_from_inputs(inputs)
        type_label = get_type_label(score_int)

        pdf_buf = build_pdf_report(inputs, score_int, type_label, tol_min, tol_max, conf, warnings)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BCS_single_report_{stamp}.pdf"

        return send_file(
            pdf_buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return render_template(
            "index.html",
            score=None,
            type_label=None,
            tolerance_hint="",
            error=f"PDF生成失败：{e}",
            warnings=warnings,
            confidence_text="",
            inputs=inputs
        )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


