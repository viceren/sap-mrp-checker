# -*- coding: utf-8 -*-
"""
验证码识别率评估 v2 - 用浏览器登录验证答案
- 每次获取验证码图片+uuid
- 求解后通过浏览器表单提交登录
- 根据URL是否跳转判断答案正确性
"""
import os, sys, io, base64, time, json, requests, logging
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from captcha_solver_v2 import ArithmeticCaptchaSolverV2

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,_,v = line.partition("=")
                os.environ[k.strip()] = v.strip()

USERNAME = os.environ.get("USERNAME", "admin")
PASSWORD = os.environ.get("PASSWORD", "")
API_BASE = os.environ.get("API_BASE", "http://192.168.220.90:9998")
BASE_URL = os.environ.get("BASE_URL", "http://192.168.220.90:8081")
CAPTCHA_API = f"{API_BASE}/evods/capcha/verifyImg"

logging.basicConfig(level=logging.WARNING, format='%(message)s')

SAMPLE_DIR = Path(__file__).parent / "captcha_samples"
SAMPLE_DIR.mkdir(exist_ok=True)


def fetch_captcha(session):
    resp = session.get(CAPTCHA_API, timeout=20)
    jd = resp.json()
    img_bytes = base64.b64decode(jd["data"]["code"])
    return img_bytes


def main():
    total = 50
    solver = ArithmeticCaptchaSolverV2()
    results = []

    print("=" * 70)
    print(f"验证码识别率评估 (浏览器登录验证, 样本={total})")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width":1440,"height":900})

        for i in range(total):
            # 关键: 每次清除所有cookie, 确保需要重新登录
            ctx.clear_cookies()
            page = ctx.new_page()
            try:
                page.goto(BASE_URL + "/#/dispatch/index", wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(1500)

                # 如果URL不含login且没有登录表单, 说明还在登录态, 强制跳登录页
                if "login" not in page.url.lower() and not page.is_visible("input.el-input__inner"):
                    # 强制跳转登录页
                    page.goto(BASE_URL + "/#/login", wait_until="networkidle", timeout=20000)
                    page.wait_for_timeout(1500)

                # 确认在登录页
                if not page.is_visible("input.el-input__inner"):
                    print(f"#{i+1}: 非登录页 URL={page.url[:60]}")
                    page.close()
                    continue

                # 获取验证码图片
                captcha_img = None
                for img in page.query_selector_all("img"):
                    src = img.get_attribute("src") or ""
                    if src.startswith("data:image"):
                        captcha_img = base64.b64decode(src.split(",",1)[1])
                        break
                if not captcha_img:
                    print(f"#{i+1}: 未找到验证码")
                    page.close()
                    continue

                # 保存样本
                sample_file = SAMPLE_DIR / f"cap_{i+1:03d}.png"
                with open(sample_file, "wb") as f:
                    f.write(captcha_img)

                # 求解
                result = solver.solve(captcha_img)
                if not result:
                    print(f"#{i+1}: 求解失败 (无候选) sample={sample_file.name}")
                    results.append((i+1, None, 0, "none", "no_answer", str(sample_file)))
                    page.close()
                    continue

                ans = result["answer"]
                conf = result["confidence"]
                method = result["method"]
                candidates = result.get("candidates", [ans])

                # 填表登录
                inputs = page.query_selector_all("input.el-input__inner")
                if len(inputs) < 3:
                    page.close()
                    continue
                inputs[0].fill(USERNAME)
                inputs[1].fill(PASSWORD)
                inputs[2].fill(str(ans))
                btns = page.query_selector_all("button.el-button--primary")
                if btns: btns[0].click()
                page.wait_for_timeout(2500)

                # 判断
                if "login" not in page.url.lower():
                    status = "ok"
                else:
                    status = "wrong"

                results.append((i+1, ans, conf, method, status, str(sample_file)))
                mark = "✓" if status == "ok" else "✗"
                print(f"#{i+1:3d}: ans={ans:>3} conf={conf:.2f} {mark} method={method[:35]} cands={candidates[:3]}")

            except Exception as e:
                print(f"#{i+1}: 异常 {e}")
                results.append((i+1, None, 0, "error", "error", ""))
            finally:
                page.close()

        browser.close()

    # ===== 统计 =====
    print("\n" + "=" * 70)
    print("统计分析")
    print("=" * 70)

    valid = [r for r in results if r[4] in ("ok", "wrong")]
    n = len(valid)
    if n == 0:
        print("无有效数据")
        return

    ok = sum(1 for r in valid if r[4] == "ok")
    wrong = sum(1 for r in valid if r[4] == "wrong")
    no_ans = sum(1 for r in results if r[4] == "no_answer")

    print(f"\n总样本: {len(results)}")
    print(f"有效验证: {n}")
    print(f"答案正确: {ok}  ({ok*100//n}%)")
    print(f"答案错误: {wrong}  ({wrong*100//n}%)")
    print(f"无答案: {no_ans}")

    # 置信度分布 + 准确率
    print(f"\n--- 置信度分布及准确率 ---")
    bins = [(0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        items = [r for r in valid if lo <= r[2] < hi]
        if items:
            ok_in = sum(1 for r in items if r[4] == "ok")
            print(f"  [{lo:.2f}, {hi:.2f}): 总{len(items):3d} 正确{ok_in:3d} 准确率{ok_in*100//len(items):3d}%")

    # 各方法准确率
    print(f"\n--- 各方法准确率 ---")
    method_stat = {}
    for r in valid:
        m = r[3].split(":")[0] if ":" in r[3] else r[3]
        if m not in method_stat:
            method_stat[m] = {"ok":0, "wrong":0, "conf_sum":0}
        method_stat[m][r[4]] += 1
        method_stat[m]["conf_sum"] += r[2]
    for m, s in sorted(method_stat.items(), key=lambda x: -(x[1]["ok"]+x[1]["wrong"])):
        total_m = s["ok"] + s["wrong"]
        avg_conf = s["conf_sum"] / total_m
        print(f"  {m:30s} 总{total_m:3d} 正确{s['ok']:3d} 准确率{s['ok']*100//max(total_m,1):3d}% 平均置信度{avg_conf:.2f}")

    # 错误案例
    wrong_cases = [r for r in valid if r[4] == "wrong"]
    print(f"\n--- 错误案例 ({len(wrong_cases)}个) ---")
    for r in wrong_cases[:20]:
        print(f"  #{r[0]:3d} ans={r[1]} conf={r[2]:.2f} method={r[3][:40]} sample={Path(r[5]).name if r[5] else '-'}")

    # 保存详细报告
    report_file = Path(__file__).parent / f"captcha_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "valid": n,
            "correct": ok,
            "wrong": wrong,
            "no_answer": no_ans,
            "accuracy": ok / max(n, 1),
            "results": [
                {"idx": r[0], "answer": r[1], "confidence": r[2], "method": r[3], "status": r[4], "sample": r[5]}
                for r in results
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告: {report_file}")


if __name__ == "__main__":
    main()
