from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from io import BytesIO
import datetime
import os


# =========================
# 1) 路径
# =========================
BASE_DIR = Path(__file__).resolve().parent

# ✅ 新内置通用模型
MODEL_PATH = BASE_DIR / "universal_auto_best.joblib"


# =========================
# 2) 加载通用模型包
# =========================
# 训练脚本产物一般结构：
# {
#   "model": pipeline,
#   "features": feats,
#   ...
# }
model_pack = joblib.load(MODEL_PATH)
model = model_pack["model"]

# 模型期望的英文特征顺序
# 如果包里没写，就用最常见 7 体尺顺序兜底
PREFERRED_FEATURES_EN = [
    "Withers Height", "Body Length", "Hip Height",
    "Heart Girth", "Abdominal Girth", "Rump Width", "Body Weight"
]
UNIVERSAL_FEATURES = model_pack.get("features") or PREFERRED_FEATURES_EN


# =========================
# 3) 网站输入（保持不变）
# =========================
FEATURE_KEYS = ["体斜长", "胸围", "腹围", "腰角宽", "体高"]

# 中文 -> 英文（与训练脚本映射一致）
CH2EN_WEB = {
    "体高": "Withers Height",
    "体斜长": "Body Length",
    "胸围": "Heart Girth",
    "腹围": "Abdominal Girth",
    "腰角宽": "Rump Width",
}

# 训练范围提示（仅对填写了的项做提醒）
FEATURE_MIN = {"体斜长": 114.4, "胸围": 142.0, "腹围": 153.0, "腰角宽": 31.0, "体高": 103.1}
FEATURE_MAX = {"体斜长": 173.0, "胸围": 216.0, "腹围": 248.0, "腰角宽": 61.0, "体高": 141.0}


app = Flask(__name__, template_folder="templates", static_folder="static")


# =========================
# 4) 模板输入安全填充
# =========================
def normalize_template_inputs(data_dict):
    """
    确保模板里用 inputs.get(...) 或 inputs['xxx'] 都不容易炸
    """
    out = {}
    data_dict = data_dict or {}
    for k in FEATURE_KEYS:
        v = data_dict.get(k, "")
        out[k] = v
    return out


# =========================
# 5) 业务工具函数
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


def confidence_level(conf):
    if conf is None:
        return ""
    if conf >= 0.75:
        return "高"
    if conf >= 0.55:
        return "中"
    return "低"


def parse_inputs(data_dict, min_required=2):
    """
    ✅ 允许只填任意 >=2 个指标
    data_dict: dict-like (来自 form 或 json)
    return: (inputs, errors, warnings)

    inputs 只包含用户实际填写且可转为 float 的键
    """
    inputs = {}
    errors = []
    warnings = []

    provided = 0

    for k in FEATURE_KEYS:
        raw = data_dict.get(k, None)

        # ✅ 允许为空
        if raw is None or str(raw).strip() == "":
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
        provided += 1

    if errors:
        return None, errors, warnings

    if provided < min_required:
        return None, [f"请至少填写 {min_required} 项体尺指标"], warnings

    # 仅对填写了的项做范围提示
    for k, v in inputs.items():
        if k in FEATURE_MIN and k in FEATURE_MAX:
            if v < FEATURE_MIN[k] or v > FEATURE_MAX[k]:
                warnings.append(
                    f"{k} 超出训练数据范围（{FEATURE_MIN[k]} ~ {FEATURE_MAX[k]} cm），结果需谨慎解读"
                )

    if "体高" in inputs and inputs["体高"] < 50:
        warnings.append("体高数值偏小，请确认单位为 cm")

    return inputs, errors, warnings


def build_features(inputs):
    """
    ✅ 通用模型特征组装：
    - 按 UNIVERSAL_FEATURES 的英文顺序组织
    - 网站没提供的（如 Hip Height / Body Weight）自动填 NaN
    - pipeline 内部会做缺失值处理 + 标准化 + 缺失指示
    """
    row = {f: np.nan for f in UNIVERSAL_FEATURES}

    for ch, en in CH2EN_WEB.items():
        if en in row:
            val = inputs.get(ch, np.nan)
            try:
                row[en] = float(val)
            except Exception:
                row[en] = np.nan

    X = pd.DataFrame([row], columns=UNIVERSAL_FEATURES)
    return X


def predict_from_inputs(inputs):
    X = build_features(inputs)

    pred = model.predict(X)[0]
    try:
        score_int = int(round(float(pred)))
    except Exception:
        score_int = int(pred)

    score_int = max(1, min(9, score_int))

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        except Exception:
            confidence = None

    tol_min = max(1, score_int - 1)
    tol_max = min(9, score_int + 1)

    return score_int, confidence, tol_min, tol_max


# =========================
# 6) 首页（单牛）
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    type_label = None
    tolerance_hint = ""
    error = ""
    warnings = []
    confidence_text = ""

    inputs_for_tpl = normalize_template_inputs({})

    if request.method == "POST":
        parsed_inputs, errors, warnings = parse_inputs(request.form, min_required=2)
        inputs_for_tpl = normalize_template_inputs(request.form)

        if errors:
            error = "；".join(errors)
            return render_template(
                "index.html",
                score=None,
                type_label=None,
                tolerance_hint="",
                error=error,
                warnings=warnings,
                confidence_text="",
                inputs=inputs_for_tpl
            )

        try:
            score_int, conf, tol_min, tol_max = predict_from_inputs(parsed_inputs)
            score = str(score_int)
            type_label = get_type_label(score_int)

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
        inputs=inputs_for_tpl
    )


# =========================
# 7) JSON API
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    parsed_inputs, errors, warnings = parse_inputs(data or {}, min_required=2)

    if errors:
        return jsonify({"ok": False, "errors": errors, "warnings": warnings}), 400

    try:
        score_int, conf, tol_min, tol_max = predict_from_inputs(parsed_inputs)
        return jsonify({
            "ok": True,
            "inputs": parsed_inputs,
            "score": score_int,
            "type_label": get_type_label(score_int),
            "tolerance": {"min": tol_min, "max": tol_max},
            "tolerance_prod": {"min": max(4, tol_min), "max": min(8, tol_max)},
            "confidence": conf,
            "confidence_level": confidence_level(conf) if conf is not None else None,
            "warnings": warnings
        })
    except Exception as e:
        return jsonify({"ok": False, "errors": [str(e)], "warnings": warnings}), 500


# =========================
# 8) 批量页面
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
        try:
            f.stream.seek(0)
            df = pd.read_excel(f)
        except Exception as e:
            return render_template("batch.html", error=f"文件解析失败：{e}")

    # ✅ 任意两列即可
    available = [c for c in FEATURE_KEYS if c in df.columns]
    if len(available) < 2:
        return render_template(
            "batch.html",
            error=f"至少需要包含 {', '.join(FEATURE_KEYS)} 中任意两列才能预测"
        )

    scores, types, tol_mins, tol_maxs, confs, notes = [], [], [], [], [], []

    for _, row in df.iterrows():
        row_dict = {k: row.get(k, None) for k in FEATURE_KEYS}
        parsed_inputs, errors, warnings = parse_inputs(row_dict, min_required=2)

        if errors:
            scores.append(np.nan)
            types.append("数据不足/错误")
            tol_mins.append(np.nan)
            tol_maxs.append(np.nan)
            confs.append(np.nan)
            notes.append("；".join(errors))
            continue

        try:
            score_int, conf, tol_min, tol_max = predict_from_inputs(parsed_inputs)
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

    return send_file(bio, mimetype="text/csv", as_attachment=True, download_name=filename)


# =========================
# 9) PDF：中文稳定 + BCS 对照表
# =========================
def get_pdf_cn_font():
    """
    字体策略：
    1) 优先使用项目内置字体（static/fonts）
    2) 若不存在，则使用 ReportLab 内置中文 CID 字体 STSong-Light
       ——解决 Render/Linux 以及本地缺字体导致的中文方框问题
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except Exception as e:
        raise RuntimeError("未安装 reportlab，请执行：pip install reportlab") from e

    candidates = [
        BASE_DIR / "static" / "fonts" / "SimHei.ttf",
        BASE_DIR / "static" / "fonts" / "msyh.ttc",
        BASE_DIR / "static" / "fonts" / "MicrosoftYaHei.ttf",
    ]

    for fp in candidates:
        if fp.exists():
            try:
                pdfmetrics.registerFont(TTFont("CNFont", str(fp)))
                return "CNFont"
            except Exception:
                pass

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        return "Helvetica"


def draw_bcs_reference_table(c, font_name, x, y, table_w):
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    row_h = 0.75 * cm
    col1 = 3.0 * cm
    col2 = 3.8 * cm

    headers = ["BCS区间", "体况分级", "简要解读"]
    rows = [
        ["1–3", "偏瘦型", "能量/脂肪储备不足，需关注营养供给"],
        ["4–6", "理想型", "生产性能与健康风险较平衡"],
        ["7",   "偏肥型", "脂肪沉积偏多，注意日粮与运动"],
        ["8–9", "肥胖型", "代谢与生产风险升高，建议干预管理"],
    ]

    # 表头
    c.setFillColor(colors.HexColor("#f3f7ff"))
    c.setStrokeColor(colors.HexColor("#d7e6ff"))
    c.rect(x, y - row_h, table_w, row_h, stroke=1, fill=1)

    c.setFont(font_name, 10.5)
    c.setFillColor(colors.HexColor("#1f3b63"))
    c.drawString(x + 0.2*cm, y - 0.55*cm, headers[0])
    c.drawString(x + col1 + 0.2*cm, y - 0.55*cm, headers[1])
    c.drawString(x + col1 + col2 + 0.2*cm, y - 0.55*cm, headers[2])

    # 内容
    c.setFont(font_name, 10)
    cy = y - row_h
    for r in rows:
        cy -= row_h
        c.setFillColor(colors.white)
        c.rect(x, cy, table_w, row_h, stroke=1, fill=1)

        c.setFillColor(colors.HexColor("#243b53"))
        c.drawString(x + 0.2*cm, cy + 0.22*cm, r[0])
        c.drawString(x + col1 + 0.2*cm, cy + 0.22*cm, r[1])
        c.setFillColor(colors.HexColor("#4b6b88"))
        c.drawString(x + col1 + col2 + 0.2*cm, cy + 0.22*cm, r[2])

    # 竖线
    c.line(x + col1, y - row_h*(len(rows)+1), x + col1, y)
    c.line(x + col1 + col2, y - row_h*(len(rows)+1), x + col1 + col2, y)

    return y - row_h*(len(rows)+1)


def build_pdf_report(inputs, score_int, type_label, tol_min, tol_max, conf, warnings):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    font_name = get_pdf_cn_font()
    warnings = warnings or []

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left = 2.0 * cm
    right = width - 2.0 * cm

    # 顶部浅色条
    c.setFillColor(colors.HexColor("#f3f7ff"))
    c.rect(0, height - 3.2 * cm, width, 3.2 * cm, stroke=0, fill=1)

    # Logo（有就画）
    logo_path = BASE_DIR / "static" / "logo.png"
    if logo_path.exists():
        try:
            c.drawImage(str(logo_path), right - 2.2 * cm, height - 2.8 * cm,
                        width=1.6 * cm, height=1.6 * cm, mask="auto")
        except Exception:
            pass

    # 标题
    c.setFillColor(colors.HexColor("#1f3b63"))
    c.setFont(font_name, 18)
    c.drawString(left, height - 2.2 * cm, "肉牛体况智能评价系统")

    c.setFont(font_name, 11)
    c.setFillColor(colors.HexColor("#4b6b88"))
    c.drawString(left, height - 2.9 * cm, "BCS 智能评分简报）")

    c.setFont(font_name, 9.5)
    c.setFillColor(colors.HexColor("#6b7b8c"))
    c.drawRightString(right, height - 2.9 * cm,
                      datetime.datetime.now().strftime("生成时间：%Y-%m-%d %H:%M:%S"))

    c.setStrokeColor(colors.HexColor("#d7e6ff"))
    c.line(left, height - 3.4 * cm, right, height - 3.4 * cm)

    y = height - 4.4 * cm

    # 输入体尺
    c.setFont(font_name, 12.5)
    c.setFillColor(colors.HexColor("#1f5fd6"))
    c.drawString(left, y, "输入体尺（cm）")
    y -= 0.8 * cm

    c.setFont(font_name, 11)
    c.setFillColor(colors.HexColor("#243b53"))

    # 只展示用户填写的项
    for k in FEATURE_KEYS:
        if k in inputs:
            c.drawString(left + 0.4*cm, y, f"{k}：{inputs[k]:.1f}")
            y -= 0.55 * cm

    y -= 0.1 * cm

    # 预测结果
    c.setFont(font_name, 12.5)
    c.setFillColor(colors.HexColor("#1f5fd6"))
    c.drawString(left, y, "智能预测结果")
    y -= 0.8 * cm

    c.setFont(font_name, 11.5)
    c.setFillColor(colors.HexColor("#0d2b52"))
    c.drawString(left + 0.4*cm, y, f"预测 BCS：{score_int}")
    y -= 0.55 * cm
    c.drawString(left + 0.4*cm, y, f"评分类型：{type_label}")
    y -= 0.55 * cm

    c.setFont(font_name, 11)
    c.setFillColor(colors.HexColor("#2f4f6b"))
    c.drawString(left + 0.4*cm, y, f"±1 容错范围：{tol_min} ~ {tol_max}")
    y -= 0.55 * cm
    c.drawString(left + 0.4*cm, y, f"生产口径：{max(4, tol_min)} ~ {min(8, tol_max)}")
    y -= 0.55 * cm

    if conf is not None:
        c.drawString(left + 0.4*cm, y, f"预测置信度：{confidence_level(conf)}（{conf:.2f}）")
        y -= 0.55 * cm

    y -= 0.2 * cm

    # ✅ BCS 对照表
    c.setFont(font_name, 12.5)
    c.setFillColor(colors.HexColor("#1f5fd6"))
    c.drawString(left, y, "BCS 对照区间表")
    y -= 0.4 * cm

    table_w = right - left
    y = draw_bcs_reference_table(c, font_name, left, y, table_w) - 0.6 * cm

    # 提示
    if warnings:
        c.setFont(font_name, 11.5)
        c.setFillColor(colors.HexColor("#1f5fd6"))
        c.drawString(left, y, "提示")
        y -= 0.7 * cm

        c.setFont(font_name, 10.5)
        c.setFillColor(colors.HexColor("#7a4a00"))
        for w in warnings[:6]:
            c.drawString(left + 0.4*cm, y, f"• {w}")
            y -= 0.5 * cm

    # 页脚
    c.setFont(font_name, 9)
    c.setFillColor(colors.HexColor("#8a98a8"))
    c.drawString(left, 2.0 * cm, "本简报为科研原型输出，建议结合现场经验与生产管理策略综合判断。")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


@app.route("/report", methods=["POST"])
def single_report():
    parsed_inputs, errors, warnings = parse_inputs(request.form, min_required=2)
    inputs_for_tpl = normalize_template_inputs(request.form)

    if errors:
        return render_template(
            "index.html",
            score=None,
            type_label=None,
            tolerance_hint="",
            error="；".join(errors),
            warnings=warnings,
            confidence_text="",
            inputs=inputs_for_tpl
        )

    try:
        score_int, conf, tol_min, tol_max = predict_from_inputs(parsed_inputs)
        type_label = get_type_label(score_int)

        pdf_buf = build_pdf_report(parsed_inputs, score_int, type_label, tol_min, tol_max, conf, warnings)
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
            inputs=inputs_for_tpl
        )


# =========================
# 10) Render 兼容启动
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
