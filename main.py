import json
import os
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

START_URL = os.environ["LEDGER_START_URL"]
TZ = ZoneInfo("Asia/Kolkata")

app = FastAPI(title="Acme Ledger Agent")
session = requests.Session()
client = OpenAI()

orders = []
rates = {}

REGIONS = {"Central", "East", "North", "South", "West"}
PRODUCTS = [
    "Air Fryer", "Blender", "Grinder", "Juicer",
    "Kettle", "Mixer", "Rice Cooker", "Toaster",
]

ALLOWED_METRICS = {
    "revenue",
    "refunds",
    "order_count",
    "unique_customer_count",
    "units_sold",
    "top_product",
    "product_revenue",
    "customer_spending",
    "top_customer",
    "average_order_value",
}


def parse_date(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)


def load_ledger():
    response = session.get(START_URL, timeout=15)
    response.raise_for_status()
    discovery = response.json()

    rate_response = session.get(discovery["links"]["rates"], timeout=15)
    rate_response.raise_for_status()
    exchange_rates = rate_response.json()["usd_per_unit"]

    export_response = session.get(discovery["links"]["export"], timeout=30)
    export_response.raise_for_status()

    # The export is NDJSON: one JSON object per line.
    records = [
        json.loads(line)
        for line in export_response.text.splitlines()
        if line.strip()
    ]

    # Keep the latest version of each order ID.
    latest = {}
    for record in records:
        order_id = record["id"]
        if (
            order_id not in latest
            or parse_date(record["updated_at"])
            > parse_date(latest[order_id]["updated_at"])
        ):
            latest[order_id] = record

    return list(latest.values()), exchange_rates


@app.on_event("startup")
def startup():
    global orders, rates
    try:
        orders, rates = load_ledger()
        print(f"Loaded {len(orders)} unique orders")
    except Exception as exc:
        print(f"Ledger loading failed: {exc}")
        raise


class QuestionRequest(BaseModel):
    question: str


def interpret_question(question):
    instructions = """
Convert the user's question into one JSON object with keys:
metric, region, product, customer, start_date, end_date,
status, date_field.

Allowed metrics:
revenue, refunds, order_count, unique_customer_count,
units_sold, top_product, product_revenue, customer_spending,
top_customer, average_order_value.

Use null for unspecified filters.
Dates must be YYYY-MM-DD.
For a month, start_date is its first day and end_date
is the first day of the following month.

Regions:
Central, East, North, South, West.

Products:
Air Fryer, Blender, Grinder, Juicer, Kettle, Mixer,
Rice Cooker, Toaster.

Rules:
- "How many different/unique/distinct customers" means
  unique_customer_count, not order_count.
- "How many orders" or "count orders" means order_count.
- "On average" means average_order_value.
- "Which product earned the most revenue" means top_product.
- For top_product, use revenue when the question asks which
  product earned the most revenue; use quantity when it asks
  which product sold the most units.
- Revenue counts only paid orders.
- Refund questions refer to refunded orders.
- Use created_at for orders described as placed, ordered, or created.
- Use updated_at when explicitly asked about update or
  refund-processing dates.
- Otherwise use created_at.
- Customer filters use IDs such as C1001.
Return JSON only.
"""

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=250,
    )

    query = json.loads(result.choices[0].message.content)
    q = question.casefold()

    if query.get("metric") not in ALLOWED_METRICS:
        raise ValueError("Unsupported metric")

    # Explicitly recognize questions asking which product earned
    # the most revenue. This must not be confused with units sold.
    asks_top_product = (
        bool(re.search(r"\b(which|what)\s+products?\b", q))
        and bool(re.search(r"\b(most|highest|top)\b", q))
    )
    asks_revenue = bool(
        re.search(r"\b(revenue|earned|earnings|sales value)\b", q)
    )
    asks_units = bool(
        re.search(r"\b(units|quantities|quantity)\b", q)
    )

    if asks_top_product and asks_revenue:
        query["metric"] = "top_product"
        query["top_product_basis"] = "revenue"
    elif asks_top_product and asks_units:
        query["metric"] = "top_product"
        query["top_product_basis"] = "quantity"

    # Check distinct-customer questions before generic count questions.
    customer_count_question = bool(
        re.search(r"\b(different|distinct|unique)\s+customers?\b", q)
        or re.search(
            r"\b(how many|count|number of)\s+"
            r"(different|distinct|unique)\s+customers?\b",
            q,
        )
    )

    if customer_count_question:
        query["metric"] = "unique_customer_count"
    elif re.search(r"\b(count|how many|number of)\b", q):
        query["metric"] = "order_count"
    elif re.search(
        r"\b(on average|average|average order value|"
        r"worth on average|per order)\b",
        q,
    ):
        query["metric"] = "average_order_value"

    # Detect explicitly mentioned region.
    for region in REGIONS:
        if re.search(r"\b" + re.escape(region.casefold()) + r"\b", q):
            query["region"] = region
            break

    # Detect explicitly mentioned product.
    for product in PRODUCTS:
        if product.casefold() in q:
            query["product"] = product
            break

    # Select the order status.
    if re.search(r"\b(refunded|refunds|refund)\b", q):
        query["status"] = "refunded"
    elif re.search(r"\b(paid|payment|payments)\b", q):
        query["status"] = "paid"
    elif re.search(r"\b(void|voided)\b", q):
        query["status"] = "void"
    elif query["metric"] == "refunds":
        query["status"] = "refunded"
    else:
        query["status"] = "paid"

    # Select the correct date field.
    if re.search(r"\b(placed|ordered|created)\b", q):
        query["date_field"] = "created_at"
    elif re.search(
        r"\b(updated|update date|refund processed|processed date)\b",
        q,
    ):
        query["date_field"] = "updated_at"
    elif query["metric"] == "refunds":
        query["date_field"] = "updated_at"
    else:
        query["date_field"] = "created_at"

    # Clear dates if the question doesn't specify a date period.
    has_date_period = bool(
        re.search(
            r"\b(january|february|march|april|may|june|july|"
            r"august|september|october|november|december|"
            r"20\d{2}|between|from|during|before|after)\b",
            q,
        )
    )

    if not has_date_period:
        query["start_date"] = None
        query["end_date"] = None

    if query.get("date_field") not in {"created_at", "updated_at"}:
        query["date_field"] = "created_at"

    return query


def calculate(query):
    metric = query["metric"]
    date_field = query.get("date_field", "created_at")
    required_status = query.get("status")

    if required_status is None:
        required_status = "refunded" if metric == "refunds" else "paid"

    filtered = []

    for order in orders:
        if order["status"] != required_status:
            continue

        if query.get("region") and (
            order["region"].casefold() != query["region"].casefold()
        ):
            continue

        if query.get("product") and (
            order["product"].casefold() != query["product"].casefold()
        ):
            continue

        if query.get("customer") and (
            order["customer"].casefold() != query["customer"].casefold()
        ):
            continue

        date_value = parse_date(order[date_field])

        if query.get("start_date"):
            start = parse_date(query["start_date"] + "T00:00:00")
            if date_value < start:
                continue

        if query.get("end_date"):
            end = parse_date(query["end_date"] + "T00:00:00")
            if date_value >= end:
                continue

        filtered.append(order)

    def usd(order):
        currency = order["currency"]
        if currency not in rates:
            raise ValueError(f"Unknown currency: {currency}")
        return (
            Decimal(str(order["amount"]))
            * Decimal(str(rates[currency]))
        )

    def money(value):
        return float(
            value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    if metric == "order_count":
        return len(filtered)

    if metric == "unique_customer_count":
        return len({order["customer"] for order in filtered})

    if metric == "units_sold":
        return sum(int(order["qty"]) for order in filtered)

    if metric in {"revenue", "refunds"}:
        return money(sum((usd(o) for o in filtered), Decimal("0")))

    if metric == "average_order_value":
        if not filtered:
            raise ValueError("No matching orders for average_order_value")
        total_usd = sum((usd(o) for o in filtered), Decimal("0"))
        return money(total_usd / Decimal(len(filtered)))

    if metric == "product_revenue":
        totals = {}
        for order in filtered:
            product = order["product"]
            totals[product] = (
                totals.get(product, Decimal("0")) + usd(order)
            )
        return money(max(totals.values())) if totals else 0.00

    if metric == "top_product":
        totals = {}

        for order in filtered:
            product = order["product"]
            totals[product] = (
                totals.get(product, Decimal("0")) + usd(order)
            )

        return max(totals, key=totals.get) if totals else None

    if metric in {"customer_spending", "top_customer"}:
        totals = {}
        for order in filtered:
            customer = order["customer"]
            totals[customer] = (
                totals.get(customer, Decimal("0")) + usd(order)
            )

        if not totals:
            return None if metric == "top_customer" else 0.00

        if metric == "top_customer":
            return max(totals, key=totals.get)

        if query.get("customer"):
            return money(
                totals.get(query["customer"], Decimal("0"))
            )

        return money(sum(totals.values(), Decimal("0")))

    raise ValueError("Unsupported calculation")


@app.get("/")
def health():
    return {"status": "ok", "orders_loaded": len(orders)}


@app.post("/")
def answer_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if not orders:
        raise HTTPException(503, "Ledger is not loaded")

    try:
        query = interpret_question(request.question)
        print(f"Interpreted query: {query}")
        answer = calculate(query)
        return {"answer": answer}
    except Exception as exc:
        print(f"Question failed: {exc}")
        raise HTTPException(500, "Could not answer this question")
