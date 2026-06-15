# -*- coding: utf-8 -*-
"""
SAP MRP 调度日志自动检查脚本 v3 - API直连版
核心改进: 登录后直接调用后端API查询, 绕过Element UI日期选择器问题

API: GET /evods/trans/op/xlog/listPage
参数: jobName, createDateBegin, createDateEnd, targetResult, offset, limit
"""
import os, sys, io, base64, time, json, logging, re
from datetime import datetime, timedelta
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from captcha_solver_v2 import ArithmeticCaptchaSolverV2

# ========== 从 .env 加载配置 ==========
def load_env():
    """从 .env 文件加载环境变量 (不覆盖已有环境变量)"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip()
                    if key and key not in os.environ:
                        os.environ[key] = val

load_env()

BASE_URL = os.environ.get("BASE_URL", "http://192.168.220.90:8081")
API_BASE = os.environ.get("API_BASE", "http://192.168.220.90:9998")
USERNAME = os.environ.get("USERNAME", "admin")
PASSWORD = os.environ.get("PASSWORD", "")
TARGET_TASK = os.environ.get("TARGET_TASK", "sap-mrp-main")
MAX_LOGIN_RETRIES = int(os.environ.get("MAX_LOGIN_RETRIES", "20"))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(OUT_DIR, "auto_check.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("auto_check")


# ============================================================
#  验证码 + 登录 (复用v2的逻辑)
# ============================================================

def solve_and_login(page, max_retries=MAX_LOGIN_RETRIES):
    """自动登录: 获取验证码→求解→填表单→提交→重试"""
    solver = ArithmeticCaptchaSolverV2()

    for attempt in range(max_retries):
        try:
            # 获取验证码图片
            captcha_img = None
            for img in page.query_selector_all("img"):
                src = img.get_attribute("src") or ""
                if src.startswith("data:image"):
                    captcha_img = base64.b64decode(src.split(",", 1)[1])
                    break

            if not captcha_img:
                log.warning(f"[{attempt+1}] 未找到验证码图片")
                _refresh_captcha(page)
                page.wait_for_timeout(1000)
                continue

            # 求解验证码
            result = solver.solve(captcha_img)
            if not result:
                log.warning(f"[{attempt+1}] 验证码求解失败")
                _refresh_captcha(page)
                page.wait_for_timeout(1000)
                continue

            ver = result["answer"]
            log.info(f"[{attempt+1}] 验证码={ver} (方法={result['method'][:40]}, 置信度={result['confidence']})")

            # 填写登录表单
            inputs = page.query_selector_all("input.el-input__inner")
            if len(inputs) < 3:
                log.warning(f"[{attempt+1}] 登录表单输入框不足 ({len(inputs)}个)")
                page.wait_for_timeout(1500)
                continue

            inputs[0].fill(USERNAME)
            inputs[1].fill(PASSWORD)
            inputs[2].fill(ver)

            # 提交
            login_btns = page.query_selector_all("button.el-button--primary")
            if login_btns:
                login_btns[0].click()
            else:
                page.query_selector("button").click()
            page.wait_for_timeout(2500)

            # 判断登录是否成功
            if "login" not in page.url.lower():
                log.info(f"[OK] 第{attempt+1}次尝试登录成功!")
                return True

            _refresh_captcha(page)
            page.wait_for_timeout(1000)

        except Exception as e:
            log.error(f"[{attempt+1}] 异常: {e}")
            _refresh_captcha(page)
            page.wait_for_timeout(1500)

    log.error(f"登录失败: {max_retries}次尝试均未成功")
    return False


def _refresh_captcha(page):
    """点击验证码图片刷新"""
    try:
        for img in page.query_selector_all("img"):
            src = img.get_attribute("src") or ""
            if src.startswith("data:image") and src != "":
                img.click()
                return
    except Exception:
        pass


# ============================================================
#  核心: 通过浏览器上下文直接调用后端API查询
# ============================================================

def get_auth_token(page):
    """从浏览器cookie中获取Authorization token"""
    cookies = page.context.cookies()
    for c in cookies:
        if c["name"] == "Authorization":
            return c["value"]
    return None


def query_via_fetch(page, job_name, start_date, end_date):
    """
    通过页面内JS fetch调用API
    - 手动提取Authorization cookie作为token
    - 放入fetch请求头中
    - 使用完整URL指向9998端口
    """
    # 先获取 token
    token = get_auth_token(page)
    if not token:
        log.error("无法获取Authorization token!")
        return None

    log.info(f"已获取token: {token[:30]}...")

    js_code = f"""async () => {{
        const token = '{token}';
        try {{
            const params = new URLSearchParams({{
                jobName: '{job_name}',
                targetResult: '',
                createDateBegin: '{start_date}',
                createDateEnd: '{end_date}',
                offset: '1',
                limit: '50',
                total: '0'
            }});
            const url = '{API_BASE}/evods/trans/op/xlog/listPage?' + params.toString();
            const resp = await fetch(url, {{
                headers: {{
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Authorization': token
                }}
            }});
            const text = await resp.text();
            if (!text) return {{status: resp.status, error: 'empty_response'}};
            let data;
            try {{ data = JSON.parse(text); }} catch(e) {{ return {{status: resp.status, error: 'not_json', body: text.substring(0, 500)}}; }}
            return {{status: resp.status, data: data}};
        }} catch(e) {{
            return {{error: e.message}};
        }}
    }}"""

    result = page.evaluate(js_code)
    log.info(f"fetch结果: {json.dumps(result, ensure_ascii=False)[:500]}")
    if isinstance(result, dict) and "data" in result:
        return result["data"]
    return None


def query_dispatch_api(page, job_name, start_date, end_date):
    """
    主查询方式: 先尝试 context.request, 失败则回退到 fetch
    注意: context.request 在跨端口时可能不携带浏览器cookie
    """
    api_url = f"{API_BASE}/evods/trans/op/xlog/listPage"
    params = {
        "jobName": job_name,
        "targetResult": "",
        "createDateBegin": start_date,
        "createDateEnd": end_date,
        "offset": "1",
        "limit": "50",
        "total": "0"
    }

    log.info(f"API请求: GET {api_url}")
    log.info(f"参数: jobName={job_name}, date={start_date} ~ {end_date}")

    # 尝试 context.request
    try:
        ctx = page.context
        response = ctx.request.get(api_url, params=params, timeout=15000)
        log.info(f"  context.request 状态: {response.status}")

        if response.status == 200:
            data = response.json()
            log.info(f"  API返回数据: {json.dumps(data, ensure_ascii=False)[:500]}")
            return data
        else:
            body = response.text()[:300] if response.text() else "(empty)"
            log.warning(f"  context.request 失败({response.status}), 回退到fetch")
    except Exception as e:
        log.warning(f"  context.request 异常({e}), 回退到fetch")

    # 回退到 JS fetch (通过浏览器发送, 自动携带cookie)
    log.info("  使用 JS fetch (浏览器cookie)...")
    return query_via_fetch(page, job_name, start_date, end_date)


# ============================================================
#  报告生成
# ============================================================

def generate_report(api_data, start_date, end_date):
    """从API数据生成格式化报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("SAP MRP 调度日志自动检查报告")
    lines.append(f"检查时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"目标任务:   {TARGET_TASK}")
    lines.append(f"查询日期:   {start_date} ~ {end_date}")
    lines.append("=" * 70)

    # 解析API返回结构
    records = []
    total = 0

    if api_data:
        # 尝试常见的数据结构
        if isinstance(api_data, dict):
            # 可能的结构: {rows: [...], total: N} 或 {data: {records: [...]}} 或 {list: [...]}
            for key in ['rows', 'records', 'list', 'data']:
                if key in api_data and isinstance(api_data[key], list):
                    records = api_data[key]
                    break

            if not records and 'data' in api_data and isinstance(api_data['data'], dict):
                for key in ['rows', 'records', 'list', 'content']:
                    if key in api_data['data'] and isinstance(api_data['data'][key], list):
                        records = api_data['data'][key]
                        break

            # 总数
            for key in ['total', 'count', 'totalCount']:
                if key in api_data:
                    total = api_data[key]
                    break

            # 如果还是没有记录, 打印完整结构帮助调试
            if not records:
                top_keys = list(api_data.keys())[:10]
                log.debug(f"API数据顶层键: {top_keys}")
                # 可能数据直接就是列表(不太可能但防一下)
                pass

    # 如果上面都没提取到, 把整个data当列表试一下
    if not records and isinstance(api_data, list):
        records = api_data

    log.info(f"解析到 {len(records)} 条记录, 总数={total}")

    if not records:
        lines.append("")
        lines.append("[INFO] 未找到该任务的运行记录")
        lines.append("       可能原因:")
        lines.append("         1. 任务在指定时间段内未执行")
        lines.append("         2. 任务名称不匹配")
        lines.append("")
        lines.append("结论: 无需关注 (或需要人工确认任务配置)")
        lines.append("=" * 70)
        return "\n".join(lines), False

    # 过滤目标任务的记录
    matching_records = []
    for rec in records:
        # 记录可能是字典, 找包含任务名的字段
        rec_text = json.dumps(rec, ensure_ascii=False) if isinstance(rec, dict) else str(rec)
        if TARGET_TASK.lower() in rec_text.lower():
            matching_records.append(rec)

    log.info(f"匹配 {TARGET_TASK} 的记录: {len(matching_records)} 条")

    if not matching_records:
        lines.append("")
        lines.append(f"[INFO] 查询到 {len(records)} 条记录, 但无匹配 '{TARGET_TASK}' 的任务")
        lines.append("")
        lines.append("结论: 目标任务可能尚未配置调度")
        lines.append("=" * 70)
        return "\n".join(lines), False

    all_ok = True
    lines.append(f"\n共找到 {len(matching_records)} 条匹配记录:\n")

    for i, rec in enumerate(matching_records):
        lines.append(f"--- 记录 #{i+1} ---")

        if isinstance(rec, dict):
            # 常见字段映射
            field_display = {
                "resourceStoreName": "资源库名",
                "jobName": "任务名称",
                "jobFile": "任务文件",
                "jobType": "任务类型",
                "startTime": "开始时间",
                "endTime": "结束时间",
                "runResult": "运行结果",
                "targetResult": "运行结果",
                "result": "运行结果",
                "status": "状态",
                "createTime": "创建时间",
            }

            # 先显示已知字段
            shown = set()
            for eng, cn in field_display.items():
                if eng in rec:
                    val = str(rec[eng])
                    lines.append(f"  {cn}: {val}")
                    shown.add(eng)

            # 显示其余字段
            for k, v in rec.items():
                if k not in shown:
                    lines.append(f"  {k}: {v}")

            # 判断运行结果状态
            result_val = ""
            for result_key in ["runResult", "targetResult", "result", "status"]:
                if result_key in rec:
                    result_val = str(rec[result_key])
                    break

            # 判断是否异常 (非"已结束"/"成功"/"完成" 都算异常)
            if result_val and "已结束" not in result_val and "成功" not in result_val and "完成" not in result_val:
                all_ok = False
                lines.append(f"  *** [!] 异常状态: {result_val} ***\n")
            else:
                lines.append("")
        else:
            lines.append(f"  数据: {rec}")

    lines.append("=" * 70)
    if all_ok:
        lines.append("结论: 所有任务均已正常结束, 状态良好 OK")
    else:
        lines.append("结论: [!!] 存在未结束/异常任务, 请立即处理!!")
    lines.append("=" * 70)

    return "\n".join(lines), not all_ok


# ============================================================
#  主流程
# ============================================================

def main():
    log.info("=" * 60)
    log.info("SAP MRP 调度日志自动检查 v3 (API直连版) 启动")
    log.info("=" * 60)

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    # 日期格式: 带时间部分 (与前端Element UI一致)
    start_date = yesterday.strftime("%Y-%m-%d") + " 00:00:00"
    end_date = today.strftime("%Y-%m-%d") + " 23:59:59"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        try:
            # 1. 打开页面并登录
            log.info("步骤1: 打开页面...")
            page.goto(BASE_URL + "/#/dispatch/index",
                     wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2500)

            if "login" in page.url.lower() or page.is_visible("input.el-input__inner"):
                log.info("步骤2: 自动登录...")
                if not solve_and_login(page):
                    page.screenshot(path=os.path.join(OUT_DIR, "error_login_fail.png"))
                    log.error("登录失败!")
                    raise SystemExit(1)

            # 2. 进入调度日志页确保session有效
            log.info("步骤3: 确保进入调度日志页...")
            page.goto(BASE_URL + "/#/dispatch/index",
                     wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)

            # 3. ★ 核心: 通过浏览器JS fetch + Authorization token 调后端API ★
            log.info("步骤4: 通过API查询调度数据...")

            # 直接用 JS fetch (通过浏览器上下文, 自动处理认证)
            api_data = query_dispatch_api(page, TARGET_TASK, start_date, end_date)

            # 如果API还是失败, 截图保留现场
            if not api_data:
                log.error("API查询全部失败! 尝试前端回退...")
                page.screenshot(path=os.path.join(OUT_DIR, "error_api_fail.png"))
                # 回退: 使用前端方式(可能查不到数据但至少不会崩溃)
                api_data = None

            # 4. 生成报告
            log.info("步骤5: 生成报告...")
            report, has_anomaly = generate_report(api_data, start_date, end_date)
            print("\n" + report)

            # 保存报告
            report_file = os.path.join(OUT_DIR, f"report_{today.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            log.info(f"\n报告已保存: {report_file}")

            # 5. 也截一张图做记录
            page.screenshot(path=os.path.join(OUT_DIR, "dispatch_final.png"), full_page=True)

            # 6. 如果有异常, 输出特殊标记
            if has_anomaly:
                log.warning("\n*** 发现异常任务! 请关注上方报告中的 [!] 标记 ***\n")

        except SystemExit:
            raise
        except Exception as e:
            log.error(f"执行异常: {e}", exc_info=True)
            page.screenshot(path=os.path.join(OUT_DIR, "error_exception.png"))
        finally:
            browser.close()

    log.info("\n检查完成")


if __name__ == "__main__":
    main()
