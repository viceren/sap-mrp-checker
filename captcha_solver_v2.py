# -*- coding: utf-8 -*-
"""
算术验证码求解器 v2
核心策略: 红色通道提取 + 干扰线去除 + 列投影分割 + 单字符/整式OCR

验证码特征 (来自诊断分析):
- 尺寸: 160×60px
- 格式: [数字]+[数字]=?  (仅加法)
- 字符颜色: 红/粉色, R>180, G≈110-130, B≈120-140
- 干扰线: 底部横线(y>50), 右侧竖线(x>145), 顶部细线(y<3)
- 字符位置: 数字1≈x[28,52], +=≈x[45,63], 数字2≈x[63,90], ==≈x[88,115], ?≈x[113,128]
"""
import io, re, base64, logging
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class ArithmeticCaptchaSolverV2:
    def __init__(self, ocr_default=None, ocr_beta=None):
        # 延迟加载 ddddocr (避免 import 时耗时)
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
        return np.array(im, dtype=np.int32), im.size  # (H,W,3), (W,H)

    def _extract_red_mask(self, arr, threshold="medium"):
        """
        提取红色/粉色字符像素的二值掩码
        threshold: "loose" / "medium" / "strict"
        """
        H, W = arr.shape[:2]
        mask = np.zeros((H, W), dtype=bool)

        if threshold == "loose":
            cond = lambda r, g, b: r > 120 and (r - g) > 25 and (r - b) > 25
        elif threshold == "strict":
            cond = lambda r, g, b: r > 170 and (r - g) > 60 and (r - b) > 55
        else:  # medium
            cond = lambda r, g, b: r > 150 and (r - g) > 45 and (r - b) > 40

        for y in range(H):
            for x in range(W):
                r, g, b = int(arr[y, x, 0]), int(arr[y, x, 1]), int(arr[y, x, 2])
                if cond(r, g, b):
                    mask[y, x] = True
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
        out = Image.new("L", (W, H), 255)
        for y in range(H):
            for x in range(W):
                if mask[y, x]:
                    out.putpixel((x, y), 0)
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

    def _parse_arithmetic(self, text):
        """解析OCR文本为算术表达式，返回 (answer, confidence)"""
        if not text:
            return None, 0
        text = text.strip()
        logger.debug(f"  [parse] OCR原文: {text!r}")

        # 清理常见误识
        cleaned = text
        replacements = [
            ('t', '+'), ('T', '+'), ('f', '+'), ('十', '+'), ('x', '+'), ('X', '+'),
            ('*', '+'),  # * 常被误识为 +
            ('一', '-'), ('_', '-'), ('~', '='), ('=', '='),
            ('i', '1'), ('l', '1'), ('I', '1'), ('|', '1'),
            ('o', '0'), ('O', '0'), ('D', '0'), ('Q', '0'),
            ('S', '5'), ('s', '5'), ('Z', '2'), ('z', '2'),
            ('B', '8'), ('g', '9'), ('a', '4'), ('A', '4'),
        ]
        for wrong, right in replacements:
            cleaned = cleaned.replace(wrong, right)
        logger.debug(f"  [parse] 清理后: {cleaned!r}")

        # 模式1: 找到 + 号, 两侧数字
        # 关键约束: 每个操作数是个位数(0-9), 答案0-18
        m = re.search(r'(\d+)\s*\+\s*(\d+)', cleaned)
        if m:
            n1_str, n2_str = m.group(1), m.group(2)
            # OCR可能把多个字符拼一起, 取合理的一位数字
            n1 = int(n1_str[-1]) if len(n1_str) > 1 else int(n1_str)
            n2 = int(n2_str[0]) if len(n2_str) > 1 else int(n2_str)
            ans = n1 + n2
            if 0 <= ans <= 18:
                exact = len(n1_str) == 1 and len(n2_str) == 1
                logger.debug(f"  [parse] 模式1: {n1}+{n2}={ans} (from {n1_str}+{n2_str})")
                return str(ans), 0.9 if exact else 0.6

        # 模式2: 有+号但正则没匹配, 手动分割
        if '+' in cleaned:
            parts = cleaned.split('+')
            if len(parts) >= 2:
                d1_list = re.findall(r'\d', parts[0])
                d2_list = re.findall(r'\d', parts[1])
                if d1_list and d2_list:
                    n1 = int(d1_list[-1])
                    n2 = int(d2_list[0])
                    ans = n1 + n2
                    if 0 <= ans <= 18:
                        logger.debug(f"  [parse] 模式2: {n1}+{n2}={ans}")
                        return str(ans), 0.7

        # 模式3: 没有识别到+号, 从所有数字中取前两位
        all_digits = re.findall(r'\d', cleaned)
        if len(all_digits) >= 2:
            n1 = int(all_digits[0])
            n2 = int(all_digits[1])
            ans = n1 + n2
            if 0 <= ans <= 18:
                logger.debug(f"  [parse] 模式3: {n1}+{n2}={ans} (digits={all_digits})")
                return str(ans), 0.4

        # 模式4: 只有一个数字
        if len(all_digits) == 1:
            n = int(all_digits[0])
            logger.debug(f"  [parse] 模式4: 单数字 {n}")
            return str(n), 0.2

        return None, 0

    # ========== 主求解方法 ==========

    def solve(self, raw_image_bytes):
        """
        主求解入口
        raw_image_bytes: 验证码图片原始字节
        返回: {"answer": "11", "method": "...", "confidence": 0.9} 或 None
        """
        arr, (W, H) = self._to_array(raw_image_bytes)

        # ===== 策略1: 严格红色提取 + 干扰去除 + 全图OCR =====
        for thresh_name in ["medium", "strict", "loose"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            img = self._mask_to_image(clean_mask, scale=4)
            img_bytes = self._image_to_bytes(img)
            candidates = self._ocr_image(img_bytes)
            logger.debug(f"  [策略1-{thresh_name}] OCR: {candidates}")

            for text in candidates:
                ans, conf = self._parse_arithmetic(text)
                if ans and conf >= 0.5:
                    return {"answer": ans, "method": f"red_clean_ocr({thresh_name}):{text!r}", "confidence": conf}

        # ===== 策略2: 列投影分割 + 单字符OCR =====
        for thresh_name in ["medium", "strict"]:
            mask = self._extract_red_mask(arr, thresh_name)
            clean_mask = self._remove_noise_lines(mask)
            col_proj = self._column_projection(clean_mask)
            regions = self._find_char_regions(col_proj)

            logger.debug(f"  [策略2-{thresh_name}] 分割区域: {regions}")

            if len(regions) >= 3:
                # 格式: D + D = ?
                # 取第1个和第3个区域作为两个数字
                digit_regions = []
                for ri, (xs, xe) in enumerate(regions):
                    width = xe - xs + 1
                    density = col_proj[xs:xe+1].sum() / max(width, 1)
                    # 数字特征: 较宽(>10px)且像素密度高
                    # +号: 较窄且居中
                    # =号: 很宽但密度低(两条横线)
                    # ?号: 较窄
                    digit_regions.append((ri, xs, xe, width, density))

                # 按像素密度排序，取密度最高的两个(应该是数字)
                digit_regions.sort(key=lambda d: d[4], reverse=True)
                # 两个最高密度的区域
                top2 = sorted(digit_regions[:2], key=lambda d: d[1])  # 按x位置排序

                digits_found = []
                for _, xs, xe, _, _ in top2:
                    crop_result = self._crop_region(clean_mask, xs, xe, padding=4)
                    if crop_result is None:
                        continue
                    crop_mask, _ = crop_result
                    crop_img = self._mask_to_image(crop_mask, scale=5)
                    crop_bytes = self._image_to_bytes(crop_img)
                    texts = self._ocr_image(crop_bytes)
                    logger.debug(f"    区域x=[{xs},{xe}] OCR: {texts}")
                    for t in texts:
                        # 只取数字
                        t_clean = re.sub(r'[^0-9]', '', t)
                        if t_clean and len(t_clean) <= 2:
                            digits_found.append(int(t_clean))
                            break

                if len(digits_found) == 2:
                    ans = digits_found[0] + digits_found[1]
                    if 0 <= ans <= 18:
                        return {"answer": str(ans), "method": f"segment_ocr({thresh_name}):{digits_found}", "confidence": 0.7}

        # ===== 策略3: 固定位置裁剪 + 单数字OCR =====
        # 基于诊断分析的字符位置范围
        for digit_x_range in [(25, 55), (60, 95)]:
            x1, x2 = digit_x_range
            sub_mask = arr[8:48, x1:x2+1]  # y范围也裁剪
            # 提取红色
            d_mask = np.zeros(sub_mask.shape[:2], dtype=bool)
            for y in range(d_mask.shape[0]):
                for x in range(d_mask.shape[1]):
                    r, g, b = int(sub_mask[y, x, 0]), int(sub_mask[y, x, 1]), int(sub_mask[y, x, 2])
                    if r > 150 and (r - g) > 40 and (r - b) > 35:
                        d_mask[y, x] = True
            # 去噪
            d_mask = self._remove_noise_lines(d_mask)
            img = self._mask_to_image(d_mask, scale=5)
            img_bytes = self._image_to_bytes(img)
            texts = self._ocr_image(img_bytes)
            logger.debug(f"  [策略3] 区域x=[{x1},{x2}] OCR: {texts}")

        # 固定位置策略: 分别OCR两个数字位置
        digit1_text = None
        digit2_text = None
        for x1, x2 in [(25, 55)]:
            sub = arr[8:48, x1:x2+1]
            d_mask = np.zeros(sub.shape[:2], dtype=bool)
            for y in range(sub.shape[0]):
                for x in range(sub.shape[1]):
                    r, g, b = int(sub[y, x, 0]), int(sub[y, x, 1]), int(sub[y, x, 2])
                    if r > 150 and (r - g) > 40 and (r - b) > 35:
                        d_mask[y, x] = True
            d_mask = self._remove_noise_lines(d_mask)
            if d_mask.sum() > 10:
                img = self._mask_to_image(d_mask, scale=5)
                texts = self._ocr_image(self._image_to_bytes(img))
                for t in texts:
                    t_clean = re.sub(r'[^0-9]', '', t)
                    if t_clean and 0 <= int(t_clean) <= 9:
                        digit1_text = int(t_clean)
                        break

        for x1, x2 in [(60, 95)]:
            sub = arr[8:48, x1:x2+1]
            d_mask = np.zeros(sub.shape[:2], dtype=bool)
            for y in range(sub.shape[0]):
                for x in range(sub.shape[1]):
                    r, g, b = int(sub[y, x, 0]), int(sub[y, x, 1]), int(sub[y, x, 2])
                    if r > 150 and (r - g) > 40 and (r - b) > 35:
                        d_mask[y, x] = True
            d_mask = self._remove_noise_lines(d_mask)
            if d_mask.sum() > 10:
                img = self._mask_to_image(d_mask, scale=5)
                texts = self._ocr_image(self._image_to_bytes(img))
                for t in texts:
                    t_clean = re.sub(r'[^0-9]', '', t)
                    if t_clean and 0 <= int(t_clean) <= 9:
                        digit2_text = int(t_clean)
                        break

        if digit1_text is not None and digit2_text is not None:
            ans = digit1_text + digit2_text
            return {"answer": str(ans), "method": f"fixed_pos_ocr:{digit1_text}+{digit2_text}", "confidence": 0.6}

        # ===== 策略4: 原图直接OCR =====
        # 放大后直接OCR
        im = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
        W0, H0 = im.size
        im_large = im.resize((W0 * 4, H0 * 4), Image.LANCZOS)
        large_bytes = self._image_to_bytes(im_large)
        candidates = self._ocr_image(large_bytes)
        logger.debug(f"  [策略4-原图] OCR: {candidates}")

        for text in candidates:
            ans, conf = self._parse_arithmetic(text)
            if ans and conf >= 0.3:
                return {"answer": ans, "method": f"raw_ocr:{text!r}", "confidence": conf}

        logger.debug("  所有策略均失败")
        return None


# ===== 测试入口 =====
if __name__ == "__main__":
    import sys, requests
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    API = "http://192.168.220.90:9998/evods/capcha/verifyImg"
    solver = ArithmeticCaptchaSolverV2()

    print("=" * 60)
    print("算术验证码求解器 v2 测试")
    print("=" * 60)

    success = 0
    total = 10
    for i in range(total):
        print(f"\n--- #{i} ---")
        try:
            resp = requests.get(API, timeout=20)
            jd = resp.json()
            raw = base64.b64decode(jd["data"]["code"])
            uuid = jd["data"]["uuid"]

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
