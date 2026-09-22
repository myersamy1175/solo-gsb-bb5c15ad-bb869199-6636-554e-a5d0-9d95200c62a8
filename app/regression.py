"""过敏原回归矩阵：同一配方代码跨版本的过敏原演变分析。

流程
----
1. 取同一 ``code`` 下的全部配方版本，按 **版本号自然序 → 创建时间 → id**
   稳定排序（数字段按数值比较，如 1.2 < 1.10；同版本号按创建先后，id
   仅作确定性兜底）。
2. 逐版复用 :func:`graph.analyze_recipe` 的替代图分析结果（直接过敏原、
   含替代链的全部过敏原、路径、循环 / 缺失标记）。
3. 对每个相邻版本对 (v[i-1] → v[i]) 输出：
   - 直接过敏原 / 替代链过敏原的新增与消失（新增项标注**首次出现版本**，
     若该过敏原曾在更早版本出现、上一版又没有，则标记为 ``regression``
     回归）；
   - 受影响配料（新增 / 移除 / 用量变化 / 过敏原集合变化 / 携带差异过敏原
     / 含循环或缺失路径的配料），并保留每条相关**来源路径**与循环、缺失、
     超深标记，供前端展开追溯。

注意：图（原料库与替代边）始终取**当前**数据对所有历史版本统一分析，
与单次对比 / 分析接口的语义一致。
"""
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from . import graph

# ---------------- 版本号稳定排序 ----------------

_TOKEN_RE = re.compile(r"\d+|[^\d]+")


def version_sort_key(version: str) -> Tuple[Tuple[int, int, str], ...]:
    """把版本字符串拆成可比较的段：数字段按数值比，文本段按字典序比。

    例："1.2" < "1.10" < "1.10.1"；"2.0-beta" < "2.0"（同前缀短者在前，
    因为前若干段相等后短元组更小）。
    """
    key: List[Tuple[int, int, str]] = []
    for tok in _TOKEN_RE.findall((version or "").strip()):
        if tok.isdigit():
            key.append((0, int(tok), ""))
        else:
            key.append((1, 0, tok))
    return tuple(key)


def list_versions_stable(conn: sqlite3.Connection, code: str):
    """按版本号 → 创建时间 → id 稳定排序返回同一代码的全部配方版本。"""
    rows = conn.execute(
        "SELECT * FROM recipes WHERE code = ? ORDER BY id", (code,)
    ).fetchall()
    # Python 的 sort 稳定；id 作为末位键仅保证任何情况下结果确定
    rows.sort(key=lambda r: (version_sort_key(r["version"]), r["created_at"], r["id"]))
    return rows


# ---------------- 辅助 ----------------

def _version_brief(order_index: int, row, analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "order_index": order_index,
        "id": row["id"],
        "version": row["version"],
        "name": row["name"],
        "created_at": row["created_at"],
        "direct_allergens": analysis["direct_allergens"],
        "chain_allergens": analysis["total_allergens"],
        "has_cycles": analysis["has_cycles"],
        "cycle_count": len(analysis["cycles"]),
        "missing_nodes": analysis["missing_nodes"],
    }


def _first_seen_info(row, order_index: int) -> Dict[str, Any]:
    return {
        "version": row["version"],
        "order_index": order_index,
        "created_at": row["created_at"],
        "recipe_id": row["id"],
    }


def _collect_sources(analysis: Dict[str, Any], allergen: str) -> List[Dict[str, Any]]:
    """汇总某过敏原在该版分析中的全部来源路径（取自 allergen_index）。"""
    sources: List[Dict[str, Any]] = []
    for root in analysis["roots"]:
        for ref in root.get("allergen_index", {}).get(allergen, []):
            sources.append(
                {
                    "ingredient_code": root["ingredient_code"],
                    "ingredient_name": root["ingredient_name"],
                    "path_id": ref["path_id"],
                    "chain": ref["chain"],
                    "introduced_by": ref["introduced_by"],
                    "via_substitution": ref["via_substitution"],
                }
            )
    return sources


def _added_entry(
    allergen: str,
    analysis: Dict[str, Any],
    order_index: int,
    first_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """新增过敏原条目：带来源路径、首次出现版本与回归标记。"""
    first = first_map.get(allergen)
    return {
        "allergen": allergen,
        "sources": _collect_sources(analysis, allergen),
        "first_seen_version": first["version"] if first else None,
        "first_seen_order_index": first["order_index"] if first else None,
        "first_seen_created_at": first["created_at"] if first else None,
        # 首次出现就在本版：系列全新；首次出现早于本版：消失后再次出现 = 回归
        "new_in_series": first is not None and first["order_index"] == order_index,
        "regression": first is not None and first["order_index"] < order_index,
    }


def _removed_entry(
    allergen: str,
    analysis: Dict[str, Any],
    first_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """消失过敏原条目：来源路径取自上一版（该过敏原最后存在的版本）。"""
    first = first_map.get(allergen)
    return {
        "allergen": allergen,
        "sources": _collect_sources(analysis, allergen),
        "first_seen_version": first["version"] if first else None,
        "first_seen_order_index": first["order_index"] if first else None,
    }


def _ordered_diff(new: List[str], old: List[str]) -> List[str]:
    old_set = set(old)
    return [a for a in new if a not in old_set]


def _measure_change(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> str:
    """同一配料代码在两版间的行级变化类型。a=旧版行，b=新版行。"""
    if a is None:
        return "added"
    if b is None:
        return "removed"
    qty_changed = a["quantity"] != b["quantity"]
    unit_changed = (a["unit"] or "") != (b["unit"] or "")
    if qty_changed and unit_changed:
        return "measure_changed"
    if qty_changed:
        return "quantity_changed"
    if unit_changed:
        return "unit_changed"
    return "unchanged"


def _relevant_paths(root: Dict[str, Any], changed_set: Set[str]) -> List[Dict[str, Any]]:
    """挑出需要随配料保留的路径：携带差异过敏原的，以及带循环/缺失/超深标记的。"""
    return [
        p
        for p in root["paths"]
        if p["cyclic"]
        or p["missing"]
        or p["truncated"]
        or (set(p["allergens"]) & changed_set)
    ]


def _ingredient_entry(
    root: Dict[str, Any],
    side: str,
    counterpart: Optional[Dict[str, Any]],
    carried: Set[str],
    changed_set: Set[str],
) -> Dict[str, Any]:
    other_direct = counterpart["direct_allergens"] if counterpart else []
    other_chain = counterpart["total_allergens"] if counterpart else []
    return {
        "side": side,  # current=新版本行 / previous=上一版行
        "ingredient_code": root["ingredient_code"],
        "ingredient_name": root["ingredient_name"],
        "exists": root["exists"],
        "quantity": root["quantity"],
        "unit": root["unit"],
        "item_change": _measure_change(counterpart, root) if side == "current"
        else _measure_change(root, counterpart),
        "direct_allergens": root["direct_allergens"],
        "chain_allergens": root["total_allergens"],
        "direct_added": _ordered_diff(root["direct_allergens"], other_direct),
        "direct_removed": _ordered_diff(other_direct, root["direct_allergens"]),
        "chain_added": _ordered_diff(root["total_allergens"], other_chain),
        "chain_removed": _ordered_diff(other_chain, root["total_allergens"]),
        # 该配料承载了本转换中哪些"新增/消失"过敏原
        "transition_allergens_carried": [a for a in root["total_allergens"] if a in carried]
        + [a for a in other_chain if a in carried and a not in set(root["total_allergens"])],
        "has_cycle": root["has_cycle"],
        "path_count": root["path_count"],
        "paths": _relevant_paths(root, changed_set),
    }


def _affected_ingredients(
    prev: Dict[str, Any],
    cur: Dict[str, Any],
    added_set: Set[str],
    removed_set: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """挑出受本次版本转换影响的配料，分新版本侧 / 上一版侧。"""
    prev_by_code = {r["ingredient_code"]: r for r in prev["roots"]}
    cur_by_code = {r["ingredient_code"]: r for r in cur["roots"]}

    current: List[Dict[str, Any]] = []
    for root in cur["roots"]:
        code = root["ingredient_code"]
        other = prev_by_code.get(code)
        changed = set(root["direct_allergens"]) != set(other["direct_allergens"]) if other else True
        changed = changed or (
            other is not None and set(root["total_allergens"]) != set(other["total_allergens"])
        )
        carries_added = bool(added_set & set(root["total_allergens"]))
        marked = root["has_cycle"] or any(p["missing"] or p["truncated"] for p in root["paths"])
        if other is None or changed or carries_added or marked:
            current.append(
                _ingredient_entry(root, "current", other, added_set, added_set | removed_set)
            )

    previous: List[Dict[str, Any]] = []
    for root in prev["roots"]:
        code = root["ingredient_code"]
        other = cur_by_code.get(code)
        changed = set(root["direct_allergens"]) != set(other["direct_allergens"]) if other else True
        changed = changed or (
            other is not None and set(root["total_allergens"]) != set(other["total_allergens"])
        )
        carries_removed = bool(removed_set & set(root["total_allergens"]))
        marked = root["has_cycle"] or any(p["missing"] or p["truncated"] for p in root["paths"])
        if other is None or changed or carries_removed or marked:
            previous.append(
                _ingredient_entry(root, "previous", other, removed_set, added_set | removed_set)
            )

    return {"current": current, "previous": previous}


# ---------------- 矩阵组装 ----------------

def build_matrix(conn: sqlite3.Connection, recipe_code: str) -> Optional[Dict[str, Any]]:
    """生成某配方代码的过敏原回归矩阵；代码不存在（无任何版本）返回 None。"""
    rows = list_versions_stable(conn, recipe_code)
    if not rows:
        return None

    analyses: List[Dict[str, Any]] = [graph.analyze_recipe(conn, r["id"]) for r in rows]

    # 过敏原在整个版本系列中的首次出现（直接 / 替代链分别记录）
    first_direct: Dict[str, Dict[str, Any]] = {}
    first_chain: Dict[str, Dict[str, Any]] = {}
    for idx, (row, an) in enumerate(zip(rows, analyses)):
        for a in an["direct_allergens"]:
            first_direct.setdefault(a, _first_seen_info(row, idx))
        for a in an["total_allergens"]:
            first_chain.setdefault(a, _first_seen_info(row, idx))

    transition_rows: List[Dict[str, Any]] = []
    for i in range(1, len(rows)):
        prev, cur = analyses[i - 1], analyses[i]

        d_added = _ordered_diff(cur["direct_allergens"], prev["direct_allergens"])
        d_removed = _ordered_diff(prev["direct_allergens"], cur["direct_allergens"])
        t_added = _ordered_diff(cur["total_allergens"], prev["total_allergens"])
        t_removed = _ordered_diff(prev["total_allergens"], cur["total_allergens"])

        added_set = set(d_added) | set(t_added)
        removed_set = set(d_removed) | set(t_removed)

        transition_rows.append(
            {
                "order_index": i,
                "from": {
                    "id": rows[i - 1]["id"],
                    "version": rows[i - 1]["version"],
                    "name": rows[i - 1]["name"],
                    "created_at": rows[i - 1]["created_at"],
                },
                "to": {
                    "id": rows[i]["id"],
                    "version": rows[i]["version"],
                    "name": rows[i]["name"],
                    "created_at": rows[i]["created_at"],
                },
                "direct": {
                    "added": [_added_entry(a, cur, i, first_direct) for a in d_added],
                    "removed": [_removed_entry(a, prev, first_direct) for a in d_removed],
                },
                "chain": {
                    "added": [_added_entry(a, cur, i, first_chain) for a in t_added],
                    "removed": [_removed_entry(a, prev, first_chain) for a in t_removed],
                },
                "affected_ingredients": _affected_ingredients(
                    prev, cur, added_set, removed_set
                ),
                "cycles": {"from": prev["cycles"], "to": cur["cycles"]},
                "missing_nodes": {
                    "from": prev["missing_nodes"],
                    "to": cur["missing_nodes"],
                },
                "has_changes": bool(
                    d_added or d_removed or t_added or t_removed
                ),
            }
        )

    all_allergens = graph._union([an["total_allergens"] for an in analyses])

    allergen_first_seen: Dict[str, Dict[str, Any]] = {}
    for allergen in all_allergens:
        fd, fc = first_direct.get(allergen), first_chain.get(allergen)
        allergen_first_seen[allergen] = {
            "first_direct_version": fd["version"] if fd else None,
            "first_direct_order_index": fd["order_index"] if fd else None,
            "first_chain_version": fc["version"] if fc else None,
            "first_chain_order_index": fc["order_index"] if fc else None,
        }

    n = len(rows)
    message = None
    if n < 2:
        message = (
            f"配方代码 {recipe_code} 目前只有 1 个版本"
            f"（v{rows[0]['version']}），至少需要 2 个版本才能生成过敏原回归矩阵"
        )

    return {
        "recipe_code": recipe_code,
        "version_count": n,
        "enough_versions": n >= 2,
        "message": message,
        "sort_rule": "版本号自然序（数字段按数值比较）→ 创建时间升序 → id 兜底",
        "versions": [_version_brief(i, r, an) for i, (r, an) in enumerate(zip(rows, analyses))],
        "rows": transition_rows,
        "allergen_first_seen": allergen_first_seen,
        "all_allergens": all_allergens,
    }
