from math import isfinite
from typing import Any

from langchain_core.tools import tool


UNIT_FACTORS = {
    "g": ("mass", 1.0),
    "kg": ("mass", 1000.0),
    "ml": ("volume", 1.0),
    "l": ("volume", 1000.0),
    "cl": ("volume", 10.0),
    "count": ("count", 1.0),
    "unit": ("count", 1.0),
    "piece": ("count", 1.0),
}

DEFAULT_WEIGHTS = {
    "unit_price": 0.55,
    "total_price": 0.20,
    "ingredients": 0.15,
    "size": 0.10,
}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _benefit_scores(values: list[float | None], lower_is_better: bool) -> list[float]:
    available = [value for value in values if value is not None]
    if not available:
        return [0.5] * len(values)

    minimum = min(available)
    maximum = max(available)
    if maximum == minimum:
        return [1.0 if value is not None else 0.5 for value in values]

    scores = []
    for value in values:
        if value is None:
            scores.append(0.5)
            continue
        normalized = (value - minimum) / (maximum - minimum)
        scores.append(1.0 - normalized if lower_is_better else normalized)
    return scores


def _normalized_weights(weights: dict[str, float] | None) -> dict[str, float]:
    provided = weights or {}
    resolved = {}
    for key, default in DEFAULT_WEIGHTS.items():
        provided_value = _number(provided.get(key))
        resolved[key] = max(
            0.0,
            default if provided_value is None else provided_value,
        )
    total = sum(resolved.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in resolved.items()}


def analyze_products(
    products: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Rank product evidence and identify potentially misleading discounts."""
    prepared = []
    for index, product in enumerate(products):
        name = str(product.get("name") or f"Product {index + 1}").strip()
        current_price = _number(product.get("current_price"))
        original_price = _number(product.get("original_price"))
        size_value = _number(product.get("size_value"))
        size_unit = str(product.get("size_unit") or "").strip().lower()
        ingredient_count = _number(product.get("ingredient_count"))
        ingredients = product.get("ingredients")
        if ingredient_count is None and isinstance(ingredients, list):
            ingredient_count = float(
                len([item for item in ingredients if str(item).strip()])
            )
        elif ingredient_count is None and isinstance(ingredients, str):
            ingredient_count = float(
                len([item for item in ingredients.split(",") if item.strip()])
            )
        claimed_discount = _number(product.get("claimed_discount_percent"))

        unit_kind = None
        base_size = None
        if size_value and size_value > 0 and size_unit in UNIT_FACTORS:
            unit_kind, factor = UNIT_FACTORS[size_unit]
            base_size = size_value * factor

        unit_price = None
        if current_price is not None and current_price >= 0 and base_size:
            multiplier = 1000.0 if unit_kind in {"mass", "volume"} else 1.0
            unit_price = current_price / base_size * multiplier

        computed_discount = None
        flags = []
        if original_price is not None and current_price is not None:
            if original_price <= 0:
                flags.append("Original price is not positive; discount cannot be verified.")
            else:
                computed_discount = (original_price - current_price) / original_price * 100
                if current_price >= original_price:
                    flags.append("The current price is not lower than the original price.")
                if computed_discount > 90:
                    flags.append("The discount exceeds 90% and should be verified.")

        if claimed_discount is not None and computed_discount is not None:
            if abs(claimed_discount - computed_discount) > 2:
                flags.append(
                    "The claimed discount differs from the calculated discount "
                    "by more than 2 percentage points."
                )

        prepared.append(
            {
                "name": name,
                "current_price": current_price,
                "original_price": original_price,
                "currency": str(product.get("currency") or "").upper(),
                "size_value": size_value,
                "size_unit": size_unit,
                "base_size": base_size,
                "unit_kind": unit_kind,
                "unit_price": unit_price,
                "ingredient_count": ingredient_count,
                "computed_discount_percent": computed_discount,
                "flags": flags,
                "source": product.get("source"),
            }
        )

    if len(prepared) < 2:
        return {
            "status": "insufficient_data",
            "message": "At least two product records are required for comparison.",
            "products": prepared,
        }

    currencies = {
        product["currency"] for product in prepared if product["currency"]
    }
    unit_kinds = {
        product["unit_kind"] for product in prepared if product["unit_kind"]
    }
    comparable_prices = len(currencies) <= 1
    comparable_unit_types = len(unit_kinds) <= 1

    unit_prices = [product["unit_price"] for product in prepared]
    total_prices = [product["current_price"] for product in prepared]
    ingredient_counts = [product["ingredient_count"] for product in prepared]
    sizes = [product["base_size"] for product in prepared]

    if not comparable_prices and (
        resolved_weights["unit_price"] > 0
        or resolved_weights["total_price"] > 0
    ):
        unit_prices = [None] * len(prepared)
        total_prices = [None] * len(prepared)
    if not comparable_unit_types and resolved_weights["unit_price"] > 0:
        unit_prices = [None] * len(prepared)

    resolved_weights = _normalized_weights(weights)
    metric_values = {
        "unit_price": unit_prices,
        "total_price": total_prices,
        "ingredients": ingredient_counts,
        "size": sizes,
    }
    missing_required_metrics = [
        metric
        for metric, metric_weight in resolved_weights.items()
        if metric_weight > 0
        and sum(value is not None for value in metric_values[metric]) < 2
    ]
    if missing_required_metrics:
        return {
            "status": "insufficient_data",
            "message": (
                "A reliable ranking cannot be calculated because fewer than two "
                "products have comparable values for: "
                + ", ".join(missing_required_metrics)
                + "."
            ),
            "missing_required_metrics": missing_required_metrics,
            "weights": resolved_weights,
            "products": prepared,
        }

    metric_scores = {
        "unit_price": _benefit_scores(unit_prices, lower_is_better=True),
        "total_price": _benefit_scores(total_prices, lower_is_better=True),
        "ingredients": _benefit_scores(ingredient_counts, lower_is_better=True),
        "size": _benefit_scores(sizes, lower_is_better=False),
    }

    ranking = []
    for index, product in enumerate(prepared):
        score = sum(
            resolved_weights[metric] * metric_scores[metric][index]
            for metric in resolved_weights
        )
        product["score"] = round(score, 4)
        if product["unit_price"] is not None:
            unit_label = (
                "kg" if product["unit_kind"] == "mass"
                else "L" if product["unit_kind"] == "volume"
                else "item"
            )
            product["unit_price_label"] = (
                f'{product["unit_price"]:.2f} '
                f'{product["currency"] or "currency units"}/{unit_label}'
            )
        else:
            product["unit_price_label"] = "Unavailable"
        ranking.append(product)

    ranking.sort(key=lambda product: product["score"], reverse=True)
    warnings = []
    if not comparable_prices:
        warnings.append(
            "Price scoring was disabled because the records use different currencies."
        )
    if not comparable_unit_types:
        warnings.append(
            "Unit-price scoring was disabled because mass, volume, and count "
            "packages cannot be directly compared."
        )
    if (
        resolved_weights["unit_price"] > 0
        and all(product["unit_price"] is None for product in prepared)
    ):
        warnings.append(
            "Unit-price scoring used neutral values because package sizes were unavailable."
        )

    return {
        "status": "completed",
        "recommended_product": ranking[0]["name"],
        "weights": resolved_weights,
        "ranking": ranking,
        "warnings": warnings,
    }


@tool
def product_decision_analyzer(
    products: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Rank products by value, price, size, and ingredient simplicity.

    The tool normalizes package sizes, calculates unit prices, checks discount
    arithmetic, and returns an explainable weighted ranking. Each product may
    provide ingredient_count or an ingredients list/string. Product facts must
    come from retrieved evidence; the tool does not search for products. It
    refuses to rank when weighted metrics lack two comparable values.
    """
    return analyze_products(products, weights)
