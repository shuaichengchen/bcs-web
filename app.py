from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
import os
import datetime

# =========================
# Flask
# =========================
app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = Path(__file__).resolve().parent

# =========================
# 前端字段（已改为“十字部高”）
# =========================
FEATURE_KEYS = ["体斜长", "胸围", "腹围", "腰角宽", "体高", "十字部高", "体重"]

# 中文 -> 英文（用于兼容训练时用英文列名的模型）
CH2EN_WEB = {
    "体高": "Withers Height",
    "体斜长": "Body Length",
    "十字部高": "Hip Height",
    "胸围": "Heart Girth",
    "腹围": "Abdominal Girth",
    "腰角宽": "Rump Width",
    "体重": "Body Weight",
}
EN2CH_WEB = {v: k for k, v in CH2EN_WEB.items()}

# 仅用于提示（可按你训练集修改）
RANGE_HINT = {
    "体斜长": (50, 160),
    "胸围": (60, 180),
    "腹围": (50, 160),
    "腰角宽": (30, 90),
    "体高": (50, 160),
    "十字部高": (50, 160),
    "体重": (20, 900),
}

# =========================
# 模型加载（兼容多种 joblib 结构）
# =========================
MODEL_PATH = BASE_DIR / "universal_auto_best.joblib"
obj = joblib.load(MODEL_PATH)

if isinstance(obj, dict):
    model = obj.get("model") or obj.get("estimator") or obj
    scaler = obj.get("scaler")  # 可能不存在（你的就是这种情况）
    features_used = obj.get("features_used")
else:
    model = obj
    scaler = None
    features_used = None

def resolve_feature_order():
    """
    优先使用包内 features_used
    否则使用 sklearn 的 feature_names_in_
    再否则退回前端字段映射后的英文顺序
    """
    if features_used:
        return list(features_used)

    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    # 默认用英文顺序
    return [CH2EN_WEB[k] for k in FEATURE_KEYS]

FEATURE_ORDER = resolve_feature_order()

# =========================
# 工具函数
# =========================
def parse_inputs(data_dict):
    """从表单/JSON 中提取用户输入"""
    inputs = {}
    for k in FEATURE_KEYS:
        val = data_dict.get(k)
        if val not in [None, ""]:
            try:
                inputs[k] = float(val)
            except ValueError:
                continue
    return inputs

def build_feature_vector(inputs: dict):
    """
    按模型特征顺序组装向量
    - 支持模型特征名是中文或英文
    """
    x = []
    for feat in FEATURE_ORDER:
        # 1) 如果模型直接用中文列名
        if feat in inputs:
            x.append(inputs[feat])
            continue

        # 2) 如果模型用英文列名
        cn = EN2CH_WEB.get(feat)
        if cn and cn in inputs:
            x.append(inputs[cn])
        else:
            x.append(np.nan)

    x = np.array(x, dtype=float).reshape(1, -1)

    # 简单缺失填充：用该行已有值均值
    if np.isnan(x).any():
        row = x[0]
        valid = row[~np.isnan(row)]
        fill = float(valid.mean()) if valid.size > 0 else 0.0
        row[np.isnan(row)] = fill
        x[0] = row

    return x

def predict_value(inputs: dict):
    """至少填 2 项才预测"""
    if len(inputs) < 2:
        return None, "请至少输入两个体尺指标。"

    x = build_feature_vector(inputs)

    # 如有 scaler 则先标准化
    if scaler is not None:
        try:
            x = scaler.transform(x)
        except Exception:
            # 有些 joblib 里 scaler 可能不是你想的对象
            pass

    try:
        y = model.predict(x)[0]
    except Exception as e:
        return None, f"模型预测失败：{e}"

    # 统一转成 float 便于显示
    try:
        y = float(y)
    except Exception:
        pass

    return y, None

def make_warnings(inputs: dict):
    warns = []
    for k, v in inputs.items():
        if k in RANGE_HINT:
            lo, hi = RANGE_HINT[k]
            if v < lo or v > hi:
                unit = "kg" if k == "体重" else "cm"
                warns.append(f"{k} 超出训练数据范围（{lo} - {hi} {unit}），结果需谨慎解读")
    return warns

def normalize_template_inputs(inputs):
    """让页面回显用户输入"""
    d = {k: "" for k in FEATURE_KEYS}
    d.update(inputs)
    return d

# =========================
# 路由：首页
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    inputs = {}
    warnings = []

    if request.method == "POST":
        inputs = parse_inputs(request.form)
        warnings = make_warnings(inputs)
        result, error = predict_value(inputs)

    # ✅ 关键：模板统一用 result
    return render_template(
        "index.html",
        result=result,
        error=error,
        warnings=warnings,
        inputs=normalize_template_inputs(inputs),
    )

# =========================
# JSON API
# =========================
@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    inputs = parse_inputs(data)
    result, error = predict_value(inputs)
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"prediction": result})

# =========================
# 批量预测
# =========================
@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "未选择文件", 400

        # 兼容 csv / excel
        filename = (file.filename or "").lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        df_out = df.copy()
        preds = []

        for _, row in df.iterrows():
            row_inputs = {}
            for k in FEATURE_KEYS:
                val = row.get(k, "")
                if pd.notna(val) and val != "":
                    try:
                        row_inputs[k] = float(val)
                    except Exception:
                        pass

            pred, err = predict_value(row_inputs)
            preds.append(pred if err is None else np.nan)

        df_out["预测结果"] = preds

        out = BytesIO()
        df_out.to_excel(out, index=False)
        out.seek(0)
        return send_file(
            out,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="批量预测结果.xlsx",
        )

    return render_template("batch.html")

# =========================
# 导出 PDF 简报
# =========================
@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    # 仅作为简版输出
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    form = request.form.to_dict()
    inputs = parse_inputs(form)
    result, error = predict_value(inputs)
    warnings = make_warnings(inputs)

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    logo_path = BASE_DIR / "static" / "logo.png"
    if logo_path.exists():
        story.append(Image(str(logo_path), width=72, height=72))
        story.append(Spacer(1, 10))

    story.append(Paragraph("肉牛体况智能评价简报", styles["Title"]))
    story.append(Spacer(1, 8))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"生成时间：{now}", styles["Normal"]))
    story.append(Spacer(1, 10))

    rows = [["指标", "数值"]]
    for k in FEATURE_KEYS:
        v = form.get(k, "")
        if v != "":
            unit = "kg" if k == "体重" else "cm"
            rows.append([k, f"{v} {unit}"])

    if result is not None:
        rows.append(["预测结果", f"{result}"])
    if error:
        rows.append(["错误", error])

    table = Table(rows, colWidths=[120, 200])
    table.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "HeiseiMin-W3"),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.black),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(table)

    if warnings:
        story.append(Spacer(1, 10))
        story.append(Paragraph("提示：", styles["Heading3"]))
        for w in warnings:
            story.append(Paragraph(f"• {w}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="预测简报.pdf")

# =========================
# 启动
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
