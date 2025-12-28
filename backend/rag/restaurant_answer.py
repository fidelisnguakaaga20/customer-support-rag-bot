import re


def safe_refusal():
    return (
        "I don’t have that information yet. "
        "Please tell me the exact item + quantity and your delivery location in Abuja, "
        "and I’ll confirm for you."
    )


# -------------------------
# Intent detection
# -------------------------
def is_menu_question(q: str) -> bool:
    q = q.lower()
    return any(
        x in q for x in ["menu", "what food", "what do you sell", "what do you have", "what meals"]
    )


def is_price_question(q: str) -> bool:
    q = q.lower()
    return any(x in q for x in ["how much", "price", "cost", "₦", "naira"])


def is_cod_question(q: str) -> bool:
    q = q.lower()
    return any(x in q for x in ["pay on delivery", "cash on delivery", "cod"])


def is_delivery_question(q: str) -> bool:
    q = q.lower()
    return any(x in q for x in ["deliver", "delivery", "dispatch", "send to"])


def is_opening_hours_question(q: str) -> bool:
    q = q.lower()
    return any(
        x in q
        for x in ["open now", "are you open", "opening", "closing", "close", "working hours", "hours"]
    )


def is_cancellation_question(q: str) -> bool:
    q = q.lower()
    return any(x in q for x in ["cancel", "cancellation", "refund"])


def is_special_diet_question(q: str) -> bool:
    q = q.lower()
    return any(
        x in q for x in ["special diet", "allergy", "gluten", "diabetic", "keto", "vegetarian", "vegan"]
    )


def is_availability_question(q: str) -> bool:
    q = q.lower()
    return any(x in q for x in ["available", "availability", "in stock", "do you have", "is chicken available"])


# -------------------------
# Evidence parsing helpers
# -------------------------
def _evidence_text(evidence_chunks: list[dict]) -> str:
    return "\n".join([c["text"] for c in evidence_chunks])


def extract_menu(evidence_chunks: list[dict]) -> list[str]:
    text = _evidence_text(evidence_chunks)
    m = re.search(
        r"\nMENU\n(.+?)(?:\n\nPRICING|\n\nAVAILABILITY|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []
    block = m.group(1).strip()
    items: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("-"):
            items.append(line.lstrip("-").strip())
    return items


def extract_opening_hours(evidence_chunks: list[dict]) -> str | None:
    """
    Robust extraction:
    1) Prefer OPENING HOURS block until next ALL-CAPS header.
    2) Fallback to WHATSAPP QUICK FAQ Q/A if present in retrieved text.
    """
    text = _evidence_text(evidence_chunks)

    # 1) Primary: OPENING HOURS block until next ALL-CAPS header
    m = re.search(
        r"\bOPENING HOURS\b\s*\n(?P<body>.*?)(?=\n[A-Z][A-Z &/()\-]{3,}\n|\Z)",
        text,
        flags=re.DOTALL,
    )
    if m:
        body = m.group("body").strip()

        lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
        bullet_lines = [ln for ln in lines if ln.lstrip().startswith("-")]
        return "\n".join(bullet_lines) if bullet_lines else body

    # 2) Backup: pull from WhatsApp Quick FAQ
    m2 = re.search(
        r"Q:\s*Are you open now\?\s*\nA:\s*(?P<ans>.*?)(?=\n\nQ:|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m2:
        return m2.group("ans").strip()

    return None


def _format_opening_hours(hours_text: str) -> str:
    """
    Prevent duplicated prefix like:
    "Our opening hours are:\nOur opening hours are Monday..."
    """
    ht = hours_text.strip()

    # If the extracted text is already a sentence that starts with "Our opening hours"
    if ht.lower().startswith("our opening hours"):
        return ht

    # If it’s bullet lines or plain lines, keep a single header
    return f"Our opening hours are:\n{ht}"


# -------------------------
# WhatsApp answer
# -------------------------
def whatsapp_style_answer(user_msg: str, evidence_chunks: list[dict]) -> str:
    q = user_msg.strip()

    # 1) Prices: ALWAYS refuse (never guess)
    if is_price_question(q):
        return (
            "Prices change regularly, so I can’t confirm a price here.\n"
            "Please tell me the item and quantity you want, and I’ll confirm the current price for you."
        )

    # 2) Menu
    if is_menu_question(q):
        items = extract_menu(evidence_chunks)
        if items:
            return "Here’s our menu:\n- " + "\n- ".join(items)
        return "Please allow me confirm our menu for you."

    # 3) Cash on delivery (MUST be checked BEFORE delivery)
    if is_cod_question(q):
        return (
            "We don’t accept cash on delivery.\n"
            "You can pay via Bank Transfer or POS payment, and we’ll confirm before delivery."
        )

    # 4) Delivery
    if is_delivery_question(q):
        if any(
            city in q.lower()
            for city in ["lagos", "ibadan", "kano", "kaduna", "port harcourt", "enugu", "jos"]
        ):
            return "Sorry, we currently deliver within Abuja only."

        return (
            "Yes, we offer delivery within Abuja.\n"
            "Please share your location (area/landmark) + the item(s) and quantity, and your preferred time."
        )

    # 5) Opening / closing hours
    if is_opening_hours_question(q):
        hours = extract_opening_hours(evidence_chunks)
        if hours:
            return _format_opening_hours(hours)

        return (
            "Our opening hours are listed in our restaurant info, but I can’t pull them right now. "
            "Please hold on while I confirm."
        )

    # 6) Cancellation
    if is_cancellation_question(q):
        return (
            "You can cancel or change an order only before preparation starts.\n"
            "Please share your order details so we can confirm the current status."
        )

    # 7) Special diet
    if is_special_diet_question(q):
        return (
            "Special dietary or allergy requests need manual confirmation.\n"
            "Please share the exact request, and we’ll confirm what we can accommodate."
        )

    # 8) Availability
    if is_availability_question(q):
        return (
            "Availability depends on stock and time.\n"
            "Please tell me the exact item and quantity, and I’ll confirm availability for you."
        )

    # Default safe WhatsApp reply
    return (
        "Please tell me what you’d like to order (item + quantity).\n"
        "If you need delivery, also share your location in Abuja and your preferred time."
    )
