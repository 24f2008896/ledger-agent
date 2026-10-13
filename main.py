
import json
import os
from datetime import datetime, timezone
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
client = OpenAI()  # Reads OPENAI_API_KEY from environment

orders = []
rates = {}


def parse_date(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)


def load_ledger():
    # The starting endpoint tells us where the export and rates live.
    response = session.get(START_URL, timeout=15)
    response.raise_for_status()
    discovery = response.json()

    rate_response = session.get(discovery["links"]["rates"], timeout=15)
    rate_response.raise_for_status()
    exchange_rates = rate_response.json()["usd_per_unit"]

    export_response = session.get(
        discovery["links"]["export"], timeout=30
    )
    export_response.raise_for_status()

    # Export is NDJSON: one JSON record per line.
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
metric, region, product, customer, start_date, end_date.

Use null for unspecified filters.
Dates must be YYYY-MM-DD.
For a month, start_date is its first day and end_date
is the first day of the following month.
Use the exact product spelling from this list:
Air Fryer, Blender, Grinder, Juicer, Kettle, Mixer,
Rice Cooker, Toaster.
Customer filters use IDs such as C1001.
Revenue counts only orders with status paid.
Refund questions refer to orders with status refunded.
Use updated_at dates for refund questions and created_at
dates for sales questions.
Choose the most appropriate allowed metric.
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
        max_tokens=200,
    )

    query = json.loads(result.choices[0].message.content)

    allowed_metrics = {
        "revenue", "refunds", "order_count", "units_sold",
        "top_product", "product_revenue", "customer_spending",
        "top_customer", "average_order_value",
    }

    if query.get("metric") not in allowed_metrics:
        raise ValueError("Unsupported metric")

    return query


def calculate(query):
    metric = query["metric"]
    filtered = []

    for order in orders:
        status = order["status"]

        # Revenue includes only paid orders.
        # Refunds use refunded orders.
        if metric == "refunds":
            if status != "refunded":
                continue
            date_field = "updated_at"
        else:
            if status != "paid":
                continue
            date_field = "created_at"

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
        return float(value.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ))

    if metric == "order_count":
        return len(filtered)

    if metric == "units_sold":
        return sum(int(o["qty"]) for o in filtered)

    if metric == "refunds":
        return money(sum((usd(o) for o in filtered), Decimal("0")))

    if metric == "revenue":
        return money(sum((usd(o) for o in filtered), Decimal("0")))

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
