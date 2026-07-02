# -*- coding: utf-8 -*-
"""
算术验证码求解器 v4 (v3 + 形态学+特征识别优化)
支持: 加(+)、减(-)、乘(×)、除(÷) 四则运算

核心策略 (v3原有):
1. 红色通道提取 + 干扰线去除
2. 固定位置分割: 数字1(x≈25-55) | 运算符(x≈45-68) | 数字2(x≈63-95) | =?(x≈88-128)
3. 运算符模板匹配 (基于像素特征区分 +、-、×、÷)
4. 多策略OCR + 加权投票
5. 整数结果约束 (÷需整除, ×结果0-81, ±结果合理范围)

v4新增优化:
6. 形态学预处理: 开运算去噪 + 闭运算连断裂笔画
7. 数字像素特征识别器: 基于0-9的宽高比/密度/对称性/连通域/空洞数做模板匹配
8. 改进投票机制: 多独立策略一致时大幅提升置信度(+0.15/策略,最高+0.4)
9. OCR与特征识别交叉验证: 对0/1/8等易混淆数字优先信任特征识别
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

    def _morphological_clean(self, mask):
        """
        形态学预处理: 开运算去噪 + 闭运算连笔
        开运算 (先腐蚀后膨胀) → 去除小噪点和细干扰线
        闭运算 (先膨胀后腐蚀) → 连接断裂的笔画
        """
        # 用 scipy 的 binary_erosion/dilation 实现
        # 结构元素: 3x3 十字型
        struct = ndimage.generate_binary_structure(2, 1)  # cross-shaped

        # 开运算: 先腐蚀再膨胀 → 去除小噪点
        opened = ndimage.binary_dilation(ndimage.binary_erosion(mask, structure=struct), structure=struct)

        # 闭运算: 再膨胀再腐蚀 → 连接断裂笔画
        closed = ndimage.binary_erosion(ndimage.binary_dilation(opened, structure=struct), structure=struct)

        return closed

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

    @staticmethod
    def _extract_operator_mask(mask, x1=40, x2=72, y1=5, y2=52):
        """裁剪运算符区域，统一供像素特征与OCR使用"""
        h, w = mask.shape
        rx1, rx2 = max(0, x1), min(w, x2)
        ry1, ry2 = max(0, y1), min(h, y2)
        return mask[ry1:ry2, rx1:rx2]

    def _score_operator_pixels(self, mask):
        """基于像素几何特征给四类运算符打分"""
        op_mask = self._extract_operator_mask(mask, x1=42, x2=70, y1=8, y2=48)
        if op_mask.sum() < 5:
            return {'+': 0.0, '-': 0.0, '×': 0.0, '÷': 0.0}

        rows = np.where(op_mask.any(axis=1))[0]
        cols = np.where(op_mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return {'+': 0.0, '-': 0.0, '×': 0.0, '÷': 0.0}

        # 先裁到实际笔画边界，减少字符偏移对中心特征的影响
        op_mask = op_mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        row_proj = op_mask.sum(axis=1)
        col_proj = op_mask.sum(axis=0)
        total_px = float(op_mask.sum())
        if total_px <= 0:
            return {'+': 0.0, '-': 0.0, '×': 0.0, '÷': 0.0}

        peak_row = int(np.argmax(row_proj))
        peak_col = int(np.argmax(col_proj))
        h_ratio = float(row_proj[peak_row]) / total_px
        v_ratio = float(col_proj[peak_col]) / total_px
        col_slice = op_mask[:, peak_col].astype(np.uint8)
        longest_v_run = 0
        current_run = 0
        for val in col_slice:
            if val:
                current_run += 1
                longest_v_run = max(longest_v_run, current_run)
            else:
                current_run = 0
        v_continuity = longest_v_run / max(op_mask.shape[0], 1)

        top_half = float(op_mask[:peak_row, :].sum())
        bot_half = float(op_mask[peak_row + 1:, :].sum())
        symmetry = 1.0 - abs(top_half - bot_half) / max(total_px, 1.0)

        diag1 = float(np.trace(op_mask.astype(np.uint8)))
        diag2 = float(np.trace(np.fliplr(op_mask).astype(np.uint8)))
        diag_ratio = (diag1 + diag2) / max(total_px, 1.0)

        labeled, n_labels = ndimage.label(op_mask)
        top_components = 0
        bot_components = 0
        for lbl in range(1, n_labels + 1):
            region = labeled == lbl
            if region.sum() < 2:
                continue
            r_idx = np.where(region.any(axis=1))[0]
            if len(r_idx) == 0:
                continue
            if r_idx[-1] < peak_row:
                top_components += 1
            elif r_idx[0] > peak_row:
                bot_components += 1

        scores = {'+': 0.0, '-': 0.0, '×': 0.0, '÷': 0.0}

        # + : 横竖主干都明显，且上下相对对称
        if h_ratio > 0.12 and v_ratio > 0.12 and v_continuity > 0.34:
            scores['+'] += min(0.85, 0.32 + h_ratio + v_ratio + 0.20 * symmetry)
        if diag_ratio < 0.38:
            scores['+'] += 0.05
        if top_components >= 1 and bot_components >= 1 and v_continuity < 0.30:
            scores['+'] *= 0.55

        # - : 横向主干明显，纵向极弱
        if h_ratio > 0.16 and v_ratio < 0.12:
            scores['-'] += min(0.90, 0.42 + h_ratio * 1.2 - v_ratio * 0.4)

        # × : 对角线占比高，横竖主干不应太强
        if diag_ratio > 0.45:
            scores['×'] += min(0.90, 0.32 + diag_ratio - 0.4 * max(h_ratio, v_ratio))

        # ÷ : 有中间横线，且上下存在独立点/小组件
        if h_ratio > 0.12 and top_components >= 1 and bot_components >= 1 and v_continuity < 0.30:
            scores['÷'] += min(0.92, 0.40 + h_ratio + 0.12 * (top_components + bot_components))

        return {op: round(min(max(score, 0.0), 0.95), 4) for op, score in scores.items()}

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
        scores = self._score_operator_pixels(mask)
        best_op = max(scores, key=scores.get)
        best_score = scores[best_op]

        if best_score < 0.2:
            return None, 0

        logger.debug(f"  运算符识别(pixel): {best_op} scores={scores}")
        return best_op, best_score

    def _score_operator_ocr(self, mask):
        """对运算符区域做多变体OCR并累计得分"""
        op_mask = self._extract_operator_mask(mask)
        if op_mask.sum() < 5:
            return {'+': 0.0, '-': 0.0, '×': 0.0, '÷': 0.0}

        op_map = {
            '+': ['+', 't', 'T', 'f', '十', 'plus'],
            '-': ['-', '一', '_', '—', 'minus'],
            '×': ['×', 'x', 'X', '*', '✕', '✖'],
            '÷': ['÷', '/', '÷', '%'],
        }

        struct = ndimage.generate_binary_structure(2, 1)
        variants = [
            ("raw", op_mask, 0.42),
            ("dilate", ndimage.binary_dilation(op_mask, structure=struct), 0.34),
            ("erode", ndimage.binary_erosion(op_mask, structure=struct), 0.24),
        ]

        scores = {'+': 0.0, '-': 0.0, '×': 0.0, '÷': 0.0}
        for _, variant_mask, variant_weight in variants:
            if variant_mask.sum() < 3:
                continue
            img = self._mask_to_image(variant_mask, scale=6)
            img_bytes = self._image_to_bytes(img)
            texts = self._ocr_image(img_bytes)
            for text in texts:
                text = text.strip()
                if not text:
                    continue
                for op, aliases in op_map.items():
                    for alias in aliases:
                        if alias in text:
                            bonus = 0.12 if text == alias or text == op else 0.0
                            scores[op] += variant_weight + bonus
                            break

        return {op: round(min(score, 0.95), 4) for op, score in scores.items()}

    def _identify_operator_ocr(self, mask):
        """OCR方式识别运算符区域"""
        scores = self._score_operator_ocr(mask)
        best_op = max(scores, key=scores.get)
        best_score = scores[best_op]
        if best_score < 0.25:
            return None, 0
        return best_op, best_score

    def _rank_operator_candidates(self, mask, arr=None):
        """融合像素特征与多变体OCR，返回排序后的运算符候选"""
        pixel_scores = self._score_operator_pixels(mask)
        ocr_scores = self._score_operator_ocr(mask)
        combined = {}
        for op in ['+', '-', '×', '÷']:
            combined[op] = round(
                min(pixel_scores.get(op, 0.0) * 0.62 + ocr_scores.get(op, 0.0) * 0.55, 0.98),
                4
            )

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] < 0.18:
            return []

        filtered = [ranked[0]]
        top_score = ranked[0][1]
        for op, score in ranked[1:]:
            if score >= 0.22 and score >= top_score * 0.58:
                filtered.append((op, score))
            if len(filtered) >= 2:
                break

        logger.debug(f"  运算符融合排序: pixel={pixel_scores} ocr={ocr_scores} ranked={filtered}")
        return filtered

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

    # ========== 数字像素特征识别器 (v3新增) ==========

    def _classify_digit_by_features(self, mask, x1, x2, y1=8, y2=48):
        """
        基于像素特征的数字分类器 (0-9)
        作为OCR的补充/验证手段
        利用: 连通域数量、宽高比、上中下三段密度、中心对称性等
        """
        H, W = mask.shape
        rx1 = max(0, x1 - 2)
        rx2 = min(W - 1, x2 + 2)
        ry1 = max(0, y1)
        ry2 = min(H - 1, y2)
        sub = mask[ry1:ry2 + 1, rx1:rx2 + 1]

        if sub.sum() < 5:
            return None

        sh, sw = sub.shape

        # 提取特征
        total_px = int(sub.sum())
        if total_px < 3:
            return None

        # 找边界框 (去除空白后)
        rows_with_px = np.where(sub.any(axis=1))[0]
        cols_with_px = np.where(sub.any(axis=0))[0]
        if len(rows_with_px) == 0 or len(cols_with_px) == 0:
            return None

        top_r, bot_r = rows_with_px[0], rows_with_px[-1]
        l_c, r_c = cols_with_px[0], cols_with_px[-1]
        char_h = bot_r - top_r + 1
        char_w = r_c - l_c + 1

        # 宽高比
        aspect_ratio = char_w / max(char_h, 1)

        # 三段密度 (上/中/下各占1/3)
        h_third = max(char_h // 3, 1)
        top_density = sub[top_r:min(top_r + h_third, bot_r + 1), :].sum() if top_r <= bot_r else 0
        mid_density = sub[min(top_r + h_third, bot_r):min(top_r + 2 * h_third, bot_r + 1), :].sum()
        bot_density = sub[min(top_r + 2 * h_third, bot_r):bot_r + 1, :].sum() if top_r <= bot_r else 0

        # 左右密度 (各半)
        w_half = max(char_w // 2, 1)
        left_density = sub[:, l_c:min(l_c + w_half, r_c + 1)].sum() if l_c <= r_c else 0
        right_density = sub[:, min(l_c + w_half, r_c):r_c + 1].sum() if l_c <= r_c else 0

        # 水平中心线像素 (中间行)
        mid_row_idx = (top_r + bot_r) // 2
        center_row_px = int(sub[mid_row_idx - ry1, :].sum()) if 0 <= mid_row_idx - ry1 < sh else 0

        # 垂直中心线像素 (中间列)
        mid_col_idx = (l_c + r_c) // 2
        center_col_px = int(sub[:, mid_col_idx - rx1].sum()) if 0 <= mid_col_idx - rx1 < sw else 0

        # 上下对称性
        symmetry_v = 1.0 - abs(top_density - bot_density) / max(total_px, 1)

        # 左右对称性
        symmetry_h = 1.0 - abs(left_density - right_density) / max(total_px, 1)

        # 连通域数量
        labeled_sub, n_labels = ndimage.label(sub)

        # 空洞数 (内部空白连通域, 不接触图像边缘的独立区域)
        inverted = ~sub
        inv_labeled, n_inv = ndimage.label(inverted)
        # 检查每个反图连通域是否完全在内部(不触碰边界) = 洞
        holes = 0
        for i in range(1, n_inv + 1):
            region_mask = (inv_labeled == i)
            if not region_mask.any():
                continue
            # 如果区域不接触任何四边 → 是空洞
            touches_edge = (region_mask[0, :].any() or region_mask[-1, :].any() or
                           region_mask[:, 0].any() or region_mask[:, -1].any())
            if not touches_edge:
                holes += 1

        # 上半部分是否有独立区域 (区分6/9 vs 8/0)
        has_top_isolated = False
        has_bot_isolated = False
        for lbl in range(1, n_labels + 1):
            region = labeled_sub == lbl
            region_rows = np.where(region.any(axis=1))[0]
            if len(region_rows) > 0 and region_rows[0] < (top_r + bot_r) // 2:
                region_top = region_rows[0]
                if region_top > top_r + char_h * 0.15:
                    has_top_isolated = True
            if len(region_rows) > 0 and region_rows[-1] > (top_r + bot_r) // 2:
                region_bot = region_rows[-1]
                if region_bot < top_r + char_h * 0.85:
                    has_bot_isolated = True

        # === 数字特征匹配 ===
        scores = {}

        for digit in range(10):
            score = self._digit_feature_score(digit, {
                'total_px': total_px, 'aspect': aspect_ratio, 'char_w': char_w, 'char_h': char_h,
                'top_d': top_density, 'mid_d': mid_density, 'bot_d': bot_density,
                'left_d': left_density, 'right_d': right_density,
                'center_row': center_row_px, 'center_col': center_col_px,
                'sym_v': symmetry_v, 'sym_h': symmetry_h,
                'n_components': n_labels, 'holes': holes,
                'has_top_iso': has_top_isolated, 'has_bot_iso': has_bot_isolated,
            }, sh, sw)
            if score > 0:
                scores[digit] = score

        if not scores:
            return None

        best_digit = max(scores, key=scores.get)
        logger.debug(f"  [特征识别] digit={best_digit} (scores={dict((d, round(s,2)) for d,s in sorted(scores.items(), key=lambda x:-x[1])[:3])})")
        return best_digit if scores[best_digit] >= 0.25 else None

    @staticmethod
    def _digit_feature_score(digit, feat, sh, sw):
        """
        计算数字 d 与给定特征的匹配得分 (0~1)
        基于 160x60 验证码中手写体数字的典型特征
        """
        s = 0.0
        a = feat['aspect']
        t, m, b = feat['top_d'], feat['mid_d'], feat['bot_d']
        sv, sh_ = feat['sym_v'], feat['sym_h']
        nc = feat['n_components']
        holes = feat['holes']

        if digit == 0:
            # 圆环: 高宽比接近1, 上下左右都对称, 有空洞
            if 0.4 <= a <= 0.95: s += 0.20
            if sv > 0.7 and sh_ > 0.65: s += 0.25
            if holes >= 1: s += 0.30  # 内部有空洞是0的关键特征
            if m > t * 0.6 and m > b * 0.6: s += 0.10  # 中间也有像素
            if nc <= 3: s += 0.05
            if t > 2 and b > 2: s += 0.08

        elif digit == 1:
            # 竖线: 很窄, 高宽比大
            if a < 0.35: s += 0.35  # 最强特征
            if a < 0.50: s += 0.15
            if feat['char_h'] > 12: s += 0.15  # 较高
            if nc <= 2: s += 0.15  # 通常一个或两个连通域
            if feat['center_col'] > 3: s += 0.10  # 中间列有较多像素

        elif digit == 2:
            # 2字形: 上多下少, 不对称
            if 0.35 <= a <= 0.85: s += 0.10
            if t > b * 1.2: s += 0.25  # 上部更密
            if sv < 0.55: s += 0.20  # 上下不对称
            if nc <= 3: s += 0.10
            if feat['right_d'] > feat['left_d'] * 0.7: s += 0.15  # 右侧有笔画

        elif digit == 3:
            # 3字形: 右侧密集, 上下都有
            if 0.40 <= a <= 0.80: s += 0.10
            if feat['right_d'] > feat['left_d']: s += 0.30  # 右偏
            if t > 2 and b > 2: s += 0.20  # 上下都有
            if sv > 0.45: s += 0.10  # 有一定对称性

        elif digit == 4:
            # 4字形: 开口向下, 上部较密, 中间有竖线
            if 0.40 <= a <= 0.90: s += 0.10
            if t > b * 1.3: s += 0.25  # 上部更密
            if m > t * 0.5: s += 0.15  # 中间有竖线
            if feat['center_col'] > 2: s += 0.10
            if nc >= 2 and nc <= 4: s += 0.10

        elif digit == 5:
            # 5字形: 上横+下半圆, 左侧上部有
            if 0.40 <= a <= 0.85: s += 0.10
            if t > b * 0.8: s += 0.15  # 顶部有横
            if feat['left_d'] > 2: s += 0.15  # 左侧有
            if b > 2: s += 0.15  # 底部有圆弧
            if sv < 0.60: s += 0.15  # 不太对称
            # 5 vs 6: 5顶部明显、底部偏圆、无洞
            if t > m * 1.2: s += 0.10
            if holes < 1: s += 0.10
            # 5 vs 3: 5上部极密
            if t > b * 1.1: s += 0.08

        elif digit == 6:
            # 6字形: 下部圆, 顶部可能有小钩
            if 0.40 <= a <= 0.85: s += 0.10
            if b > t * 1.1: s += 0.20  # 下部更密
            if feat['has_bot_iso']: s += 0.20  # 底部有封闭区域
            if holes >= 1: s += 0.15  # 有空洞
            if feat['left_d'] > feat['right_d'] * 0.5: s += 0.05
            # 6 vs 8: 6上部更稀、整体偏小
            if t < b * 0.9: s += 0.10
            if feat['char_h'] < 18: s += 0.05
            # 6 vs 5: 6底部极密
            if b > m * 1.4: s += 0.08
            # 6 vs 0: 6不对称
            if sv < 0.60: s += 0.08

        elif digit == 7:
            # 7字形: 上横+斜线, 上密下稀
            if 0.30 <= a <= 0.75: s += 0.10
            if t > b * 1.5: s += 0.30  # 明显上重下轻
            if sv < 0.40: s += 0.20  # 严重不对称
            if feat['right_d'] > feat['left_d']: s += 0.10  # 斜向右
            # 7 vs 1: 7通常更宽、上部更密
            if a > 0.45: s += 0.15
            if t > m * 1.8: s += 0.10
            # 7 vs 9: 7顶部横线明显、无封闭区
            if holes < 1: s += 0.10
            if not feat['has_top_iso']: s += 0.08

        elif digit == 8:
            # 8字形: 两个圈, 对称, 可能有2个空洞
            if 0.40 <= a <= 0.85: s += 0.10
            if sv > 0.70: s += 0.25  # 上下对称
            if t > 2 and b > 2: s += 0.15  # 上下都有
            if holes >= 1: s += 0.15  # 有空洞
            if m > t * 0.5 and m > b * 0.5: s += 0.10  # 中间连接
            if nc >= 2: s += 0.05
            # 8 vs 3: 8左右对称、上下对称
            if sh_ > 0.60: s += 0.12
            if feat['left_d'] > 2 and feat['right_d'] > 2: s += 0.08
            # 8 vs 6: 8上部密度不低
            if t > b * 0.65: s += 0.08
            # 8 vs 0: 8有中间细腰
            if m < t * 0.85 and m < b * 0.85: s += 0.06

        elif digit == 9:
            # 9字形: 上部圆, 底部可能有小尾巴
            if 0.40 <= a <= 0.85: s += 0.10
            if t > b * 1.1: s += 0.20  # 上部更密
            if feat['has_top_iso']: s += 0.20  # 顶部有封闭区域
            if holes >= 1: s += 0.15  # 有空洞
            if feat['right_d'] > feat['left_d'] * 0.5: s += 0.05

        return min(s, 1.0)

    # ========== 四则运算解析 ==========

    @staticmethod
    def _normalize_ocr_text(text):
        """清理 OCR 常见误识, 统一成更适合算式解析的文本"""
        if not text:
            return ""

        cleaned = text.strip()
        replacements = [
            # 运算符替换
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
        return cleaned

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

        cleaned = self._normalize_ocr_text(text)
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

        # ===== 策略1: 红色提取 + 形态学增强 + 全图OCR =====
        for thresh_name in ["medium", "strict", "loose"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            morphed = self._morphological_clean(clean_mask)  # 形态学增强
            img = self._mask_to_image(morphed, scale=4)
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
        # 数字1: x≈22-55, 数字2: x≈58-98 (自适应微调)
        best_digits = None
        for thresh_name in ["strict", "medium"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            # 形态学预处理: 连断裂笔画、去噪点
            morphed = self._morphological_clean(clean_mask)

            d1_ocr = self._ocr_digit(morphed, 22, 55)
            d2_ocr = self._ocr_digit(morphed, 58, 98)

            # 特征识别作为补充/验证
            d1_feat = self._classify_digit_by_features(morphed, 22, 55)
            d2_feat = self._classify_digit_by_features(morphed, 58, 98)

            # OCR优先, 特征识别作为验证/修正
            d1 = d1_ocr if d1_ocr is not None else d1_feat
            d2 = d2_ocr if d2_ocr is not None else d2_feat

            # 如果OCR和特征识别结果不同且都有值, 用特征识别验证(特征识别对某些数字更准)
            if d1_ocr is not None and d1_feat is not None and d1_ocr != d1_feat:
                # 特征识别对数字1(竖线)、0(环形)、8(双圈)更可靠
                reliable_digits = {0, 1, 8}
                if d1_feat in reliable_digits:
                    d1 = d1_feat  # 信任特征识别
                    logger.debug(f"  digit1: ocr={d1_ocr} → feat={d1_feat} (trust feat)")
            if d2_ocr is not None and d2_feat is not None and d2_ocr != d2_feat:
                reliable_digits = {0, 1, 8}
                if d2_feat in reliable_digits:
                    d2 = d2_feat
                    logger.debug(f"  digit2: ocr={d2_ocr} → feat={d2_feat} (trust feat)")

            if d1 is not None and d2 is not None:
                best_digits = (d1, d2)
                op_ranked = self._rank_operator_candidates(morphed, arr)
                if op_ranked:
                    op_order = [op for op, _ in op_ranked]
                else:
                    op_order = ['+', '-', '×', '÷']

                for idx, try_op in enumerate(op_order):
                    ans = self._compute(d1, d2, try_op)
                    if ans is not None and self._is_valid_result(d1, d2, ans):
                        op_score = dict(op_ranked).get(try_op, 0.0)
                        conf = 0.42 + op_score * 0.32
                        if idx == 0:
                            conf += 0.08
                        elif idx >= 1:
                            conf -= 0.06
                        # 如果特征识别也确认了这两个数字, 额外加分
                        feat_bonus = 0.05 if (d1_feat == d1 and d2_feat == d2) else 0
                        all_candidates.append((str(ans), conf + feat_bonus, f"fixed_pos_morph:{d1}{try_op}{d2}"))
                break  # 优先使用strict，成功后不再回退到更松阈值

        # ===== 策略3: 列投影分割 + 形态学增强 =====
        for thresh_name in ["strict", "medium"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            morphed = self._morphological_clean(clean_mask)  # 形态学增强
            col_proj = self._column_projection(morphed)
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
                    crop_result = self._crop_region(morphed, xs, xe, padding=4)
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

        # ===== 策略4: 原图直接OCR (降权) =====
        im = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
        W0, H0 = im.size
        im_large = im.resize((W0 * 4, H0 * 4), Image.LANCZOS)
        large_bytes = self._image_to_bytes(im_large)
        candidates = self._ocr_image(large_bytes)

        for text in candidates:
            ans, conf, _ = self._parse_arithmetic(text)
            if ans and conf >= 0.3:
                all_candidates.append((ans, conf * 0.7, f"raw_ocr:{text!r}"))  # 降权30%

        # ===== 策略5: 纯特征识别 (v3新增, 不依赖OCR) =====
        for thresh_name in ["strict", "medium"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            morphed = self._morphological_clean(clean_mask)

            fd1 = self._classify_digit_by_features(morphed, 22, 55)
            fd2 = self._classify_digit_by_features(morphed, 58, 98)

            # 给特征识别加持OCR结果：当OCR有结果时优先信任OCR
            cd1 = self._ocr_digit(morphed, 22, 55)
            cd2 = self._ocr_digit(morphed, 58, 98)

            # OCR和特征识别交叉确认
            if cd1 is not None and cd2 is not None:
                d1, d2 = cd1, cd2
                feat_agree = (fd1 == d1 and fd2 == d2)
                cross_bonus = 0.18 if feat_agree else 0.06
            elif cd1 is not None and fd2 is not None:
                d1, d2 = cd1, fd2
                cross_bonus = 0.05 if fd1 == d1 else 0.0
            elif fd1 is not None and cd2 is not None:
                d1, d2 = fd1, cd2
                cross_bonus = 0.05 if fd2 == d2 else 0.0
            elif fd1 is not None and fd2 is not None:
                if thresh_name != "strict":
                    continue
                d1, d2 = fd1, fd2
                cross_bonus = -0.10
            else:
                continue

            op_ranked = self._rank_operator_candidates(morphed, arr)
            op_order = [op for op, _ in op_ranked] if op_ranked else ['+', '-', '×', '÷']
            for idx, try_op in enumerate(op_order):
                ans = self._compute(d1, d2, try_op)
                if ans is not None and self._is_valid_result(d1, d2, ans):
                    op_score = dict(op_ranked).get(try_op, 0.0)
                    base_conf = 0.38 + op_score * 0.28 + cross_bonus
                    if idx == 0:
                        base_conf += 0.08
                    elif idx >= 1:
                        base_conf -= 0.05
                    all_candidates.append((str(ans), base_conf, f"feat_only:{d1}{try_op}{d2}"))
            break  # 优先使用strict，成功后不再回退到更松阈值

        # ===== P0改进: 候选过滤与降权 (2026-06-25, 基于50样本真实登录验证) =====
        # 数据: 总体准确率22%, fixed_pos_morph仅7%, 0.80-0.90区间0%, 0答案准确率约13%
        filtered = []
        for ans, conf, method in all_candidates:
            method_prefix = method.split(":")[0] if ":" in method else method
            method_base = method_prefix.split("(")[0]  # "red_clean_ocr(medium)" -> "red_clean_ocr"
            method_variant = ""
            if "(" in method_prefix and ")" in method_prefix:
                method_variant = method_prefix.split("(", 1)[1].split(")", 1)[0]

            # 规则1: fixed_pos_morph 的 "0" 答案直接丢弃 ("0陷阱", 几乎全是误判)
            if ans == "0" and method_base == "fixed_pos_morph":
                continue

            # 规则2: fixed_pos_morph 整体大幅降权 (准确率仅7%)
            if method_base == "fixed_pos_morph":
                conf *= 0.3

            # 规则2.5: 按真实准确率对不同OCR来源做基础重加权
            # strict 历史表现最好，raw/binary/segment 较差，medium 保守处理
            if method_base == "red_clean_ocr":
                if method_variant == "strict":
                    conf *= 1.18
                elif method_variant == "loose":
                    conf *= 1.05
                elif method_variant == "medium":
                    conf *= 0.88
            elif method_base == "raw_ocr":
                conf *= 0.55
            elif method_base == "binary_ocr":
                conf *= 0.60
            elif method_base == "segment_ocr":
                conf *= 0.65

            # 规则3: OCR原文字母占比过高时降权 (噪声污染, 如'8t1x'->9)
            ocr_match = re.search(r"'([^']*)'", method)
            if ocr_match and method_base in ("raw_ocr", "red_clean_ocr", "segment_ocr"):
                ocr_text = ocr_match.group(1)
                if len(ocr_text) > 0:
                    letter_ratio = sum(1 for c in ocr_text if c.isalpha()) / len(ocr_text)
                    if letter_ratio > 0.4:
                        conf *= 0.5  # 字母占比>40%降权50%

                    cleaned_ocr = self._normalize_ocr_text(ocr_text)
                    digit_count = len(re.findall(r'\d', cleaned_ocr))
                    has_operator = bool(re.search(r'[+\-×÷]', cleaned_ocr))

                    # "932" / "752" 这类纯数字串是 medium 阈值的主要误判来源
                    if not has_operator and digit_count >= 2:
                        # strict 阈值下的纯数字串仍有一定可能性正确，适度降权而不是砍掉
                        if method_base == "red_clean_ocr" and method_variant == "strict":
                            conf *= 0.65
                        else:
                            conf *= 0.40

                    # 既有运算符又混入超过2个数字，通常是多余噪声字符拼接
                    if has_operator and digit_count > 2:
                        conf *= 0.80

            # 规则4: "0"答案降权 (0+N/0×N准确率仅13%, 降权后让非0答案优先)
            if ans == "0":
                conf *= 0.6

            filtered.append((ans, conf, method))

        all_candidates = filtered

        # ===== 最终决策: 改进的加权投票 =====
        if not all_candidates:
            logger.debug("  所有策略均失败 (或被P0过滤)")
            return None

        # 按答案分组, 累加置信度
        vote = {}
        for ans, conf, method in all_candidates:
            if ans not in vote:
                vote[ans] = {"total_conf": 0, "count": 0, "methods": set(), "best_conf": conf, "best_method": method}
            vote[ans]["total_conf"] += conf
            vote[ans]["count"] += 1
            vote[ans]["methods"].add(method.split(":")[0])  # 记录独立策略来源
            if conf > vote[ans]["best_conf"]:
                vote[ans]["best_conf"] = conf
                vote[ans]["best_method"] = method

        # 选择总置信度最高的答案
        ranked_vote = sorted(vote.items(), key=lambda item: item[1]["total_conf"], reverse=True)
        best_ans, v = ranked_vote[0]

        # 多策略一致则提高置信度 (P0: 收紧加分, 避免错误答案置信度虚高)
        n_strategies = len(v["methods"])  # 不同独立策略的数量
        strategy_bonus = min(0.08 * (n_strategies - 1), 0.16)  # 每个额外策略+0.08, 最高+0.16

        final_conf = min(
            v["best_conf"] + strategy_bonus,
            0.97
        )

        logger.debug(f"  投票结果: {dict((k, round(val['total_conf'],2)) for k,val in vote.items())} -> {best_ans} (策略数={n_strategies})")

        runner_up_total = ranked_vote[1][1]["total_conf"] if len(ranked_vote) > 1 else 0.0
        vote_margin = v["total_conf"] - runner_up_total
        if len(ranked_vote) > 1:
            runner_ratio = runner_up_total / max(v["total_conf"], 1e-6)
            # 只在"双雄僵局"时才拒识，放宽margin判断
            if runner_ratio > 0.92 and vote_margin < 0.12 and final_conf < 0.68:
                logger.debug(
                    f"  拒识: best={best_ans} 与次优过近 (margin={vote_margin:.2f}, ratio={runner_ratio:.2f})"
                )
                return None

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

        # 拒识阈值: 0.38，过度限制会导致大量拒识浪费重试机会
        if final_conf < 0.38:
            logger.debug(f"  拒识: final_conf={final_conf:.2f} < 0.38, 返回None触发重试")
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
