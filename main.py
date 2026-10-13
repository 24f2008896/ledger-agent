
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
client = OpenAI()  # Uses OPENAI_API_KEY and OPENAI_BASE_URL from environment

orders = []
rates = {}


def parse_date(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)


def load_ledger():
    # Discover the ledger's export and exchange-rate endpoints.
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
Convert the user's question into one JSON object.

Allowed metrics:
revenue, refunds, order_count, units_sold,
top_product, product_revenue, customer_spending,
top_customer, average_order_value.

Return these keys:
metric, region, product, customer, start_date, end_date,
status, date_field.

Use null for unspecified filters.
Dates must be YYYY-MM-DD.
For a month, start_date is its first day and end_date
is the first day of the following month.

Use the exact product spelling from this list:
Air Fryer, Blender, Grinder, Juicer, Kettle, Mixer,
Rice Cooker, Toaster.

Customer filters use IDs such as C1001.

Important rules:
- Questions asking to count, how many, or the number of orders
  must use metric order_count, even when they mention refunds.
- Revenue counts only paid orders.
- A question about refunded orders must use status refunded.
- A question about paid orders must use status paid.
- For order-count questions, count matching orders; do not return money.
- Use created_at for orders described as placed, ordered, or created.
- Use updated_at when the question explicitly asks when an order
  was updated or when a refund was processed.
- For other refund questions without an explicit date meaning,
  use updated_at.
- Use created_at for ordinary sales questions.
- region, product, and customer should be null if unspecified.
- status must be paid, refunded, void, or null.
- date_field must be created_at or updated_at.
- Return JSON only.
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

    allowed_metrics = {
        "revenue",
        "refunds",
        "order_count",
        "units_sold",
        "top_product",
        "product_revenue",
        "customer_spending",
        "top_customer",
        "average_order_value",
    }

    if query.get("metric") not in allowed_metrics:
        raise ValueError("Unsupported metric")

    # Enforce obvious intent from the original question rather than
    # relying entirely on the model's interpretation.
    question_lower = question.casefold()

    if re.search(r"\b(count|how many|number of)\b", question_lower):
        query["metric"] = "order_count"

    # Explicit order status in the question takes priority.
    if re.search(r"\b(refunded|refunds|refund)\b", question_lower):
        query["status"] = "refunded"
    elif re.search(r"\b(paid|payments)\b", question_lower):
        query["status"] = "paid"
    elif re.search(r"\b(void|voided)\b", question_lower):
        query["status"] = "void"
    elif query.get("metric") == "refunds":
        query["status"] = "refunded"
    else:
        query["status"] = "paid"

    # "Placed in February" means the order's creation date,
    # not the date it was subsequently updated or refunded.
    if re.search(r"\b(placed|ordered|created)\b", question_lower):
        query["date_field"] = "created_at"
    elif re.search(
        r"\b(updated|update date|refund processed|processed date)\b",
        question_lower,
    ):
        query["date_field"] = "updated_at"
    elif query.get("metric") == "refunds":
        query["date_field"] = "updated_at"
    else:
        query["date_field"] = "created_at"

    if query.get("status") not in {None, "paid", "refunded", "void"}:
        raise ValueError("Unsupported order status")

    if query.get("date_field") not in {"created_at", "updated_at"}:
        query["date_field"] = "created_at"

    return query


def calculate(query):
    metric = query["metric"]
    filtered = []
    date_field = query.get("date_field", "created_at")
    required_status = query.get("status")

    if required_status is None:
        required_status = "refunded" if metric == "refunds" else "paid"

    for order in orders:
        # Apply the requested status before calculating anything.
        if order["status"] != required_status:
            continue

        if query.get("region"):
            if order["region"].casefold() != query["region"].casefold():
                continue

        if query.get("product"):
            if order["product"].casefold() != query["product"].casefold():
                continue

        if query.get("customer"):
            if order["customer"].casefold() != query["customer"].casefold():
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
            value.quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
        )

    # Counts are counts of unique, filtered order IDs.
    if metric == "order_count":
        return len(filtered)

    if metric == "units_sold":
        return sum(int(order["qty"]) for order in filtered)

    if metric == "refunds":
        return money(sum((usd(o) for o in filtered), Decimal("0")))

    if metric == "revenue":
        # Revenue must always count paid orders only.
        paid_orders = [o for o in filtered if o["status"] == "paid"]
        return money(sum((usd(o) for o in paid_orders), Decimal("0")))

    if metric == "product_revenue":
        totals = {}
        for order in filtered:
            product = order["product"]
            totals[product] = totals.get(product, Decimal("0")) + usd(order)

        if not totals:
            return 0.00

        return money(max(totals.values()))

    if metric == "top_product":
        totals = {}
        for order in filtered:
            product = order["product"]
            totals[product] = totals.get(product, 0) + int(order["qty"])

        if not totals:
            return None

        return max(totals, key=totals.get)

    if metric in {"customer_spending", "top_customer"}:
        totals = {}
        for order in filtered:
            customer = order["customer"]
            totals[customer] = totals.get(customer, Decimal("0")) + usd(order)

        if not totals:
            return None if metric == "top_customer" else 0.00

        if metric == "top_customer":
            return max(totals, key=totals.get)

        return money(sum(totals.values(), Decimal("0")))

    if metric == "average_order_value":
        if not filtered:
            return 0.00

        return money(
            sum((usd(o) for o in filtered), Decimal("0"))
            / Decimal(len(filtered))
        )

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
        answer = calculate(query)
        return {"answer": answer}
    except Exception as exc:
        print(f"Question failed: {exc}")
        raise HTTPException(500, "Could not answer this question")
