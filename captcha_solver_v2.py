# -*- coding: utf-8 -*-
"""
算术验证码求解器 v3
支持: 加(+)、减(-)、乘(×)、除(÷) 四则运算

核心策略:
1. 红色通道提取 + 干扰线去除
2. 固定位置分割: 数字1(x≈25-55) | 运算符(x≈45-68) | 数字2(x≈63-95) | =?(x≈88-128)
3. 运算符模板匹配 (基于像素特征区分 +、-、×、÷)
4. 多策略OCR + 加权投票
5. 整数结果约束 (÷需整除, ×结果0-81, ±结果合理范围)
"""
import io, re, base64, logging, math
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class ArithmeticCaptchaSolverV2:
    def __init__(self, ocr_default=None, ocr_beta=None):
        self._ocr_default = ocr_default
        self._ocr_beta = ocr_beta

    @property
    def ocr_default(self):
        if self._ocr_default is None:
            import ddddocr
            self._ocr_default = ddddocr.DdddOcr(show_ad=False)
        return self._ocr_default

    @property
    def ocr_beta(self):
        if self._ocr_beta is None:
            import ddddocr
            try:
                self._ocr_beta = ddddocr.DdddOcr(show_ad=False, beta=True)
            except Exception:
                self._ocr_beta = None
        return self._ocr_beta

    # ========== 图像预处理 ==========

    def _to_array(self, raw_bytes):
        """原始字节 → RGB numpy数组"""
        im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(im, dtype=np.int32), im.size

    def _extract_red_mask(self, arr, threshold="medium"):
        """提取红色/粉色字符像素的二值掩码"""
        r_ch = arr[:, :, 0]
        g_ch = arr[:, :, 1]
        b_ch = arr[:, :, 2]

        if threshold == "loose":
            mask = (r_ch > 120) & ((r_ch - g_ch) > 25) & ((r_ch - b_ch) > 25)
        elif threshold == "strict":
            mask = (r_ch > 170) & ((r_ch - g_ch) > 60) & ((r_ch - b_ch) > 55)
        else:  # medium
            mask = (r_ch > 150) & ((r_ch - g_ch) > 45) & ((r_ch - b_ch) > 40)

        return mask

    def _remove_noise_lines(self, mask):
        """去除干扰线 (宽+矮 或 窄+高 的组件)"""
        labeled, n = ndimage.label(mask)
        clean = mask.copy()
        for cid in range(1, n + 1):
            cmask = (labeled == cid)
            px = cmask.sum()
            if px < 3:
                clean[cmask] = False
                continue
            ys, xs = np.where(cmask)
            w = xs.max() - xs.min() + 1
            h = ys.max() - ys.min() + 1
            # 干扰线特征: 横线 (宽>40且高<8) 或 竖线 (宽<8且高>40)
            if (w > 40 and h < 8) or (w < 8 and h > 40):
                clean[cmask] = False
            # 很小的噪点
            if px < 5 and w < 5 and h < 5:
                clean[cmask] = False
        return clean

    def _mask_to_image(self, mask, scale=3):
        """二值掩码 → PIL灰度图 (放大scale倍)"""
        H, W = mask.shape
        out_arr = np.full((H, W), 255, dtype=np.uint8)
        out_arr[mask] = 0
        out = Image.fromarray(out_arr, mode="L")
        if scale > 1:
            out = out.resize((W * scale, H * scale), Image.LANCZOS)
        return out

    def _image_to_bytes(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ========== 分割策略 ==========

    def _column_projection(self, mask):
        """列投影: 每列红色像素数"""
        return mask.sum(axis=0)

    def _find_char_regions(self, col_proj, min_gap=3, min_width=5, min_density=8):
        """
        从列投影找字符区域
        返回: [(x_start, x_end), ...] 列表
        """
        W = len(col_proj)
        regions = []
        in_region = False
        start = 0
        for x in range(W):
            if col_proj[x] > 0 and not in_region:
                start = x
                in_region = True
            elif col_proj[x] == 0 and in_region:
                if x - start >= min_width:
                    regions.append((start, x - 1))
                in_region = False
        if in_region and W - start >= min_width:
            regions.append((start, W - 1))

        # 合并间隔太小的区域
        if len(regions) <= 1:
            return regions

        merged = [regions[0]]
        for r in regions[1:]:
            if r[0] - merged[-1][1] <= min_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        return merged

    def _crop_region(self, mask, x_start, x_end, padding=3):
        """裁剪字符区域, 去除空白行, 加padding"""
        H, W = mask.shape
        x1 = max(0, x_start - padding)
        x2 = min(W - 1, x_end + padding)
        sub = mask[:, x1:x2 + 1]

        # 去除空白行
        row_proj = sub.sum(axis=1)
        nonzero_rows = np.where(row_proj > 0)[0]
        if len(nonzero_rows) == 0:
            return None
        y1 = max(0, nonzero_rows[0] - padding)
        y2 = min(H - 1, nonzero_rows[-1] + padding)

        return mask[y1:y2 + 1, x1:x2 + 1], (x1, y1, x2, y2)

    # ========== 运算符识别 (核心改进) ==========

    def _identify_operator(self, mask, arr):
        """
        基于像素特征识别运算符: + - × ÷
        运算符位于 x≈45-68 区域
        
        策略:
        1. 十字交叉 → + 
        2. 水平横线 → -
        3. 交叉+斜线 → ×
        4. 横线+竖线(偏移) → ÷
        """
        H, W = mask.shape
        # 运算符区域
        op_x1, op_x2 = 42, 70
        op_y1, op_y2 = 8, 48
        op_mask = mask[op_y1:op_y2, op_x1:op_x2]
        
        if op_mask.sum() < 5:
            return None, 0
        
        # 分析像素分布
        col_proj = op_mask.sum(axis=0)  # 每列
        row_proj = op_mask.sum(axis=1)  # 每行
        
        # 水平中心线和垂直中心线
        mid_row = op_mask.shape[0] // 2
        mid_col = op_mask.shape[1] // 2
        
        # 统计关键特征
        h_pixels = row_proj[mid_row] if mid_row < len(row_proj) else 0  # 水平中心行像素数
        v_pixels = col_proj[mid_col] if mid_col < len(col_proj) else 0  # 垂直中心列像素数
        
        # 上下半部分像素分布
        top_half = op_mask[:mid_row, :].sum()
        bot_half = op_mask[mid_row:, :].sum()
        
        # 是否有斜线 (×号特征: 非中心行列也有较多像素)
        diag_pixels = op_mask.sum() - h_pixels * (op_mask.shape[1] / max(col_proj.max(), 1)) - v_pixels * (op_mask.shape[0] / max(row_proj.max(), 1))
        
        total_px = op_mask.sum()
        
        # 水平连续段数量
        h_runs = 0
        in_run = False
        for c in range(op_mask.shape[1]):
            if op_mask[mid_row, c] > 0 and not in_run:
                h_runs += 1
                in_run = True
            elif op_mask[mid_row, c] == 0:
                in_run = False
        
        # 垂直连续段数量
        v_runs = 0
        in_run = False
        for r in range(op_mask.shape[0]):
            if op_mask[r, mid_col] > 0 and not in_run:
                v_runs += 1
                in_run = True
            elif op_mask[r, mid_col] == 0:
                in_run = False
        
        # 判断逻辑
        scores = {'+': 0, '-': 0, '×': 0, '÷': 0}
        
        # +号: 水平+垂直都有明显像素, 上下对称
        if h_pixels > 3 and v_pixels > 3:
            symmetry = 1 - abs(top_half - bot_half) / max(total_px, 1)
            scores['+'] = symmetry * (h_pixels + v_pixels) / max(total_px, 1)
        
        # -号: 只有水平像素, 垂直几乎无
        if h_pixels > 3 and v_pixels <= 2:
            scores['-'] = h_pixels / max(total_px, 1)
        
        # ×号: 有斜线特征, 上下都有像素但不在正中心
        # 检查对角线方向像素
        diag1_px = 0  # 左上→右下
        diag2_px = 0  # 右上→左下
        for i in range(min(op_mask.shape)):
            if i < op_mask.shape[0] and i < op_mask.shape[1]:
                diag1_px += op_mask[i, i]
            r2 = op_mask.shape[0] - 1 - i
            if 0 <= r2 < op_mask.shape[0] and i < op_mask.shape[1]:
                diag2_px += op_mask[r2, i]
        
        if diag1_px > 5 and diag2_px > 5:
            scores['×'] = (diag1_px + diag2_px) / max(total_px * 2, 1)
        
        # ÷号: 水平线 + 上下各一个点
        # 特征: 有水平线, 但上下半部分各有独立的点状像素
        if h_pixels > 3:
            top_dots = 0
            bot_dots = 0
            for r in range(mid_row):
                if op_mask[r, :].sum() > 0 and op_mask[r, :].sum() < 6:
                    top_dots += 1
            for r in range(mid_row, op_mask.shape[0]):
                if op_mask[r, :].sum() > 0 and op_mask[r, :].sum() < 6:
                    bot_dots += 1
            if top_dots >= 1 and bot_dots >= 1 and v_pixels <= 2:
                scores['÷'] = 0.8
        
        # 选择得分最高的运算符
        best_op = max(scores, key=scores.get)
        best_score = scores[best_op]
        
        if best_score < 0.2:
            return None, 0
        
        logger.debug(f"  运算符识别: {best_op} (scores={scores}, h_px={h_pixels}, v_px={v_pixels})")
        return best_op, best_score

    def _identify_operator_ocr(self, mask):
        """OCR方式识别运算符区域"""
        H, W = mask.shape
        op_x1, op_x2 = 40, 72
        op_y1, op_y2 = 5, 52
        op_mask = mask[op_y1:op_y2, op_x1:op_x2]
        
        if op_mask.sum() < 5:
            return None, 0
        
        img = self._mask_to_image(op_mask, scale=5)
        img_bytes = self._image_to_bytes(img)
        
        results = []
        try:
            r1 = self.ocr_default.classification(img_bytes)
            if r1:
                results.append(r1)
        except Exception:
            pass
        if self.ocr_beta:
            try:
                r2 = self.ocr_beta.classification(img_bytes)
                if r2:
                    results.append(r2)
            except Exception:
                pass
        
        op_map = {
            '+': ['+', 't', 'T', 'f', '十', 'plus'],
            '-': ['-', '一', '_', '—', 'minus'],
            '×': ['×', 'x', 'X', '*', '✕', '✖'],
            '÷': ['÷', '/', '÷', '%'],
        }
        
        for text in results:
            text = text.strip()
            for op, aliases in op_map.items():
                for alias in aliases:
                    if alias in text:
                        return op, 0.7
        
        return None, 0

    # ========== OCR ==========

    def _ocr_image(self, img_bytes):
        """用多个模型OCR，返回所有候选结果"""
        results = []
        try:
            r1 = self.ocr_default.classification(img_bytes)
            if r1:
                results.append(r1)
        except Exception:
            pass
        if self.ocr_beta:
            try:
                r2 = self.ocr_beta.classification(img_bytes)
                if r2:
                    results.append(r2)
            except Exception:
                pass
        return results

    def _ocr_digit(self, mask, x1, x2, y1=8, y2=48):
        """OCR识别指定区域的单个数字"""
        H, W = mask.shape
        rx1, rx2 = max(0, x1 - 3), min(W, x2 + 3)
        ry1, ry2 = max(0, y1), min(H, y2)
        sub = mask[ry1:ry2, rx1:rx2]
        
        if sub.sum() < 5:
            return None
        
        img = self._mask_to_image(sub, scale=5)
        img_bytes = self._image_to_bytes(img)
        texts = self._ocr_image(img_bytes)
        
        for t in texts:
            t_clean = re.sub(r'[^0-9]', '', t)
            if t_clean and 0 <= int(t_clean) <= 9:
                return int(t_clean)
        
        # 放宽: 取最后一位数字
        for t in texts:
            digits = re.findall(r'\d', t)
            if digits:
                d = int(digits[-1])
                if 0 <= d <= 9:
                    return d
        
        return None

    # ========== 四则运算解析 ==========

    def _parse_arithmetic(self, text):
        """
        解析OCR文本为四则运算表达式，返回 (answer, confidence, operator)
        支持: +、-、×、÷
        约束: 操作数为个位数(0-9)
        """
        if not text:
            return None, 0, None
        text = text.strip()
        logger.debug(f"  [parse] OCR原文: {text!r}")

        # 清理常见误识
        cleaned = text
        replacements = [
            # 运算符替换 (注意: t→+ 是最常见的OCR误识)
            ('t', '+'), ('T', '+'), ('f', '+'), ('十', '+'),
            ('一', '-'), ('_', '-'), ('—', '-'),
            ('x', '×'), ('X', '×'), ('*', '×'), ('✕', '×'), ('✖', '×'),
            ('/', '÷'), ('%', '÷'),
            # 数字替换
            ('i', '1'), ('l', '1'), ('I', '1'), ('|', '1'),
            ('o', '0'), ('O', '0'), ('D', '0'), ('Q', '0'),
            ('S', '5'), ('s', '5'), ('Z', '2'), ('z', '2'),
            ('B', '8'), ('g', '9'), ('a', '4'), ('A', '4'),
        ]
        for wrong, right in replacements:
            cleaned = cleaned.replace(wrong, right)
        logger.debug(f"  [parse] 清理后: {cleaned!r}")

        # 尝试匹配 N op N 格式 (按常见度排序: + > × > - > ÷)
        op_patterns = [
            (r'(\d+)\s*[+]\s*(\d+)', '+'),
            (r'(\d+)\s*[×]\s*(\d+)', '×'),
            (r'(\d+)\s*[-]\s*(\d+)', '-'),
            (r'(\d+)\s*[÷]\s*(\d+)', '÷'),
        ]
        
        for op_pattern, op_sym in op_patterns:
            m = re.search(op_pattern, cleaned)
            if m:
                n1_str, n2_str = m.group(1), m.group(2)
                n1 = int(n1_str[-1]) if len(n1_str) > 1 else int(n1_str)
                n2 = int(n2_str[0]) if len(n2_str) > 1 else int(n2_str)
                
                if n1 > 9 or n2 > 9:
                    continue
                    
                ans = self._compute(n1, n2, op_sym)
                if ans is not None and self._is_valid_result(n1, n2, ans):
                    exact = len(n1_str) == 1 and len(n2_str) == 1
                    logger.debug(f"  [parse] 匹配: {n1}{op_sym}{n2}={ans}")
                    return str(ans), 0.9 if exact else 0.6, op_sym

        # 没有运算符: 从数字中提取
        all_digits = re.findall(r'\d', cleaned)
        ops_found = re.findall(r'[+\-×÷]', cleaned)
        
        if len(all_digits) >= 2:
            n1 = int(all_digits[0])
            n2 = int(all_digits[1])
            if n1 <= 9 and n2 <= 9:
                if ops_found:
                    op = ops_found[0]
                    ans = self._compute(n1, n2, op)
                    if ans is not None and self._is_valid_result(n1, n2, ans):
                        return str(ans), 0.5, op
                else:
                    # 默认加法
                    ans = n1 + n2
                    if self._is_valid_result(n1, n2, ans):
                        return str(ans), 0.4, '+'

        if len(all_digits) == 1:
            return str(int(all_digits[0])), 0.2, None

        return None, 0, None

    def _compute(self, n1, n2, op):
        """执行四则运算, 无效结果返回None"""
        if op == '+':
            return n1 + n2
        elif op == '-':
            r = n1 - n2
            return r if r >= 0 else None  # 验码结果通常非负
        elif op == '×':
            return n1 * n2
        elif op in ('÷', '/'):
            if n2 == 0 or n1 % n2 != 0:
                return None
            return n1 // n2
        return None

    def _is_valid_result(self, n1, n2, result, expr=""):
        """验证运算结果是否合理"""
        if result is None:
            return False
        # 结果必须是非负整数
        if not isinstance(result, int) or result < 0:
            return False
        # 合理范围: 0-81 (9×9=81)
        if result > 81:
            return False
        return True

    # ========== 主求解方法 ==========

    def solve(self, raw_image_bytes):
        """
        主求解入口
        raw_image_bytes: 验证码图片原始字节
        返回: {"answer": "11", "candidates": ["11", "5", "6", "0"], "method": "...", "confidence": 0.9} 或 None
        
        candidates: 按优先级排列的候选答案列表
        +优先(最常见), 然后×, -, ÷
        """
        arr, (W, H) = self._to_array(raw_image_bytes)
        all_candidates = []  # 收集所有候选结果用于投票

        # ===== 策略1: 红色提取 + 全图OCR =====
        for thresh_name in ["medium", "strict", "loose"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            img = self._mask_to_image(clean_mask, scale=4)
            img_bytes = self._image_to_bytes(img)
            candidates = self._ocr_image(img_bytes)
            logger.debug(f"  [策略1-{thresh_name}] OCR: {candidates}")

            for text in candidates:
                ans, conf, op = self._parse_arithmetic(text)
                if ans and conf >= 0.3:
                    all_candidates.append((ans, conf, f"red_clean_ocr({thresh_name}):{text!r}"))

        # ===== 策略1.5: 灰度+二值化预处理 + OCR (保留更多信息) =====
        im = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
        # 转灰度
        gray = im.convert("L")
        # 自适应二值化: 去掉浅色背景, 保留深色字符
        gray_arr = np.array(gray)
        # 红色字符在灰度图中值偏低 (因为R高但G/B低)
        # 用中值阈值
        median = np.median(gray_arr)
        binary = gray_arr.copy()
        binary[binary > median] = 255
        binary[binary <= median] = 0
        binary_img = Image.fromarray(binary.astype(np.uint8), mode="L")
        # 放大
        W0, H0 = binary_img.size
        binary_large = binary_img.resize((W0 * 4, H0 * 4), Image.LANCZOS)
        binary_bytes = self._image_to_bytes(binary_large)
        candidates = self._ocr_image(binary_bytes)
        logger.debug(f"  [策略1.5-二值化] OCR: {candidates}")
        for text in candidates:
            ans, conf, op = self._parse_arithmetic(text)
            if ans and conf >= 0.3:
                all_candidates.append((ans, conf, f"binary_ocr:{text!r}"))

        # ===== 策略2: 固定位置分割 + 数字OCR (核心策略) =====
        # 数字1: x≈22-55, 数字2: x≈58-98
        best_digits = None
        for thresh_name in ["medium", "strict"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)

            d1 = self._ocr_digit(clean_mask, 22, 55)
            d2 = self._ocr_digit(clean_mask, 58, 98)

            if d1 is not None and d2 is not None:
                best_digits = (d1, d2)
                # 运算符识别 (仅用于排序候选优先级)
                op_pixel, op_pixel_conf = self._identify_operator(clean_mask, arr)
                op_ocr, op_ocr_conf = self._identify_operator_ocr(clean_mask)
                
                identified_op = op_pixel if op_pixel_conf >= op_ocr_conf else op_ocr
                op_conf = max(op_pixel_conf, op_ocr_conf)
                
                # 生成4种运算的候选, 按识别到的运算符优先
                op_order = ['+', '-', '×', '÷']
                if identified_op and identified_op in op_order:
                    op_order.remove(identified_op)
                    op_order.insert(0, identified_op)
                
                for try_op in op_order:
                    ans = self._compute(d1, d2, try_op)
                    if ans is not None and self._is_valid_result(d1, d2, ans):
                        conf = 0.6 if try_op == identified_op else 0.35
                        all_candidates.append((str(ans), conf, f"fixed_pos_ocr:{d1}{try_op}{d2}"))
                break  # 只要medium成功就不用strict

        # ===== 策略3: 列投影分割 =====
        for thresh_name in ["medium", "strict"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            col_proj = self._column_projection(clean_mask)
            regions = self._find_char_regions(col_proj)

            if len(regions) >= 3:
                region_info = []
                for ri, (xs, xe) in enumerate(regions):
                    width = xe - xs + 1
                    density = col_proj[xs:xe+1].sum() / max(width, 1)
                    region_info.append((ri, xs, xe, width, density))

                by_density = sorted(region_info, key=lambda d: d[4], reverse=True)
                top2 = sorted(by_density[:2], key=lambda d: d[1])

                digits_found = []
                for _, xs, xe, _, _ in top2:
                    crop_result = self._crop_region(clean_mask, xs, xe, padding=4)
                    if crop_result is None:
                        continue
                    crop_mask, _ = crop_result
                    crop_img = self._mask_to_image(crop_mask, scale=5)
                    crop_bytes = self._image_to_bytes(crop_img)
                    texts = self._ocr_image(crop_bytes)
                    for t in texts:
                        t_clean = re.sub(r'[^0-9]', '', t)
                        if t_clean and len(t_clean) <= 2:
                            digits_found.append(int(t_clean))
                            break

                if len(digits_found) == 2:
                    for try_op in ['+', '-', '×', '÷']:
                        ans = self._compute(digits_found[0], digits_found[1], try_op)
                        if ans is not None and self._is_valid_result(digits_found[0], digits_found[1], ans):
                            op_conf = 0.5 if try_op == '+' else 0.35
                            all_candidates.append((str(ans), op_conf, f"segment_ocr({thresh_name}):{digits_found[0]}{try_op}{digits_found[1]}"))

        # ===== 策略4: 原图直接OCR =====
        im = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
        W0, H0 = im.size
        im_large = im.resize((W0 * 4, H0 * 4), Image.LANCZOS)
        large_bytes = self._image_to_bytes(im_large)
        candidates = self._ocr_image(large_bytes)

        for text in candidates:
            ans, conf, _ = self._parse_arithmetic(text)
            if ans and conf >= 0.3:
                all_candidates.append((ans, conf, f"raw_ocr:{text!r}"))

        # ===== 最终决策: 加权投票 =====
        if not all_candidates:
            logger.debug("  所有策略均失败")
            return None

        # 按答案分组, 累加置信度
        vote = {}
        for ans, conf, method in all_candidates:
            if ans not in vote:
                vote[ans] = {"total_conf": 0, "count": 0, "best_method": method, "best_conf": conf}
            vote[ans]["total_conf"] += conf
            vote[ans]["count"] += 1
            if conf > vote[ans]["best_conf"]:
                vote[ans]["best_conf"] = conf
                vote[ans]["best_method"] = method

        # 选择总置信度最高的答案
        best_ans = max(vote, key=lambda a: vote[a]["total_conf"])
        v = vote[best_ans]
        
        # 多策略一致则提高置信度
        final_conf = min(v["best_conf"] * (1 + 0.1 * v["count"]), 0.95)
        
        logger.debug(f"  投票结果: {dict((k, round(v['total_conf'],2)) for k, v in vote.items())} -> {best_ans}")

        # 生成候选答案列表 (按总置信度降序)
        sorted_answers = sorted(vote.keys(), key=lambda a: vote[a]["total_conf"], reverse=True)
        
        # 去重并限制数量
        candidate_list = []
        seen = set()
        for a in sorted_answers:
            if a not in seen and a != best_ans:
                candidate_list.append(a)
                seen.add(a)
                if len(candidate_list) >= 3:
                    break

        if final_conf < 0.2:
            return None

        return {
            "answer": best_ans,
            "candidates": [best_ans] + candidate_list,
            "method": v["best_method"],
            "confidence": round(final_conf, 2)
        }


# ===== 测试入口 =====
if __name__ == "__main__":
    import sys, requests
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    API = "http://192.168.220.90:9998/evods/capcha/verifyImg"
    solver = ArithmeticCaptchaSolverV2()

    print("=" * 60)
    print("算术验证码求解器 v3 测试 (支持加减乘除)")
    print("=" * 60)

    success = 0
    total = 15
    for i in range(total):
        print(f"\n--- #{i+1} ---")
        try:
            resp = requests.get(API, timeout=20)
            jd = resp.json()
            raw = base64.b64decode(jd["data"]["code"])

            result = solver.solve(raw)
            if result:
                print(f">>> 结果: {result['answer']} | 方法: {result['method']} | 置信度: {result['confidence']}")
                success += 1
            else:
                print(">>> 识别失败")
        except Exception as e:
            print(f"错误: {e}")

    print(f"\n{'='*60}")
    print(f"测试完成: {success}/{total} 成功率={success*100//total}%")
