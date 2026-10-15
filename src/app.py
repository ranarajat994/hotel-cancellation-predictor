# ============================================================
# Hotel Booking Cancellation Prediction - app.py
# Run from project root:  streamlit run src/app.py
# ============================================================

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import datetime
import streamlit as st

from predict import predict_booking
from predict import encoder
from predict import threshold


# ------------------------------------------------------------
# 2. Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="StayGuard · Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------
# 3. Custom styling (CSS)
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, input, button, select {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background: #f6f3ec; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }

/* ---------- Hero banner ---------- */
.hero {
    background: linear-gradient(135deg, #0f2a44 0%, #17395c 60%, #1f4a72 100%);
    border-radius: 22px;
    padding: 34px 40px;
    color: #f6f3ec;
    position: relative;
    overflow: hidden;
    margin-bottom: 26px;
}
.hero::after {
    content: "";
    position: absolute;
    right: -60px; top: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    border: 38px solid rgba(200, 164, 92, 0.18);
}
.hero-tag {
    display: inline-block;
    font-size: 12px; letter-spacing: 2px; text-transform: uppercase;
    color: #c8a45c; font-weight: 700; margin-bottom: 8px;
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 40px; line-height: 1.1; margin: 0 0 10px 0; color: #ffffff;
}
.hero p { font-size: 16px; color: #c9d5e2; margin: 0; max-width: 640px; }
.hero-stats { display: flex; gap: 34px; margin-top: 24px; flex-wrap: wrap; }
.hero-stat b { font-family: 'Fraunces', serif; font-size: 26px; color: #ffffff; display: block; }
.hero-stat span { font-size: 12px; color: #9fb3c8; text-transform: uppercase; letter-spacing: 1px; }

/* ---------- Section titles ---------- */
.section-title {
    font-family: 'Fraunces', serif;
    font-size: 21px; color: #0f2a44; margin: 6px 0 2px 0;
}
.section-sub { font-size: 13px; color: #7a7466; margin-bottom: 10px; }

/* ---------- Form card ---------- */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e7e1d4;
    border-radius: 20px;
    padding: 22px 24px 10px 24px;
    box-shadow: 0 8px 24px rgba(15, 42, 68, 0.06);
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #f6f3ec; border-radius: 10px; padding: 8px 16px; font-weight: 500;
}
.stTabs [aria-selected="true"] { background: #0f2a44 !important; color: #ffffff !important; }

/* ---------- Predict button ---------- */
[data-testid="stFormSubmitButton"] button {
    background: #c8a45c; color: #0f2a44; border: none;
    font-weight: 700; font-size: 16px; padding: 12px 0; border-radius: 12px;
    transition: all 0.2s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #0f2a44; color: #ffffff; transform: translateY(-1px);
}

/* ---------- Result card ---------- */
.result-card {
    background: #ffffff; border: 1px solid #e7e1d4; border-radius: 20px;
    padding: 26px; box-shadow: 0 8px 24px rgba(15, 42, 68, 0.06);
    text-align: center;
}
.empty-card { padding: 50px 26px; color: #7a7466; }
.empty-card .icon { font-size: 46px; margin-bottom: 10px; }

.gauge {
    width: 190px; height: 190px; border-radius: 50%;
    margin: 6px auto 18px auto;
    display: flex; align-items: center; justify-content: center;
}
.gauge-inner {
    width: 146px; height: 146px; border-radius: 50%; background: #ffffff;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.gauge-inner b { font-family: 'Fraunces', serif; font-size: 38px; color: #0f2a44; }
.gauge-inner span { font-size: 12px; color: #7a7466; text-transform: uppercase; letter-spacing: 1px; }

.verdict {
    display: inline-block; padding: 8px 18px; border-radius: 999px;
    font-weight: 700; font-size: 15px; margin-bottom: 6px;
}
.verdict-high { background: #fbe4e1; color: #b3261e; }
.verdict-mid  { background: #fdf0d9; color: #9a6400; }
.verdict-low  { background: #e2f3e7; color: #1e7a3c; }
.verdict-note { font-size: 13px; color: #7a7466; margin-top: 6px; }

/* ---------- Mini stats ---------- */
.mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 18px; }
.mini {
    background: #f6f3ec; border-radius: 12px; padding: 12px; text-align: left;
}
.mini span { font-size: 11px; color: #7a7466; text-transform: uppercase; letter-spacing: 1px; }
.mini b { display: block; font-size: 18px; color: #0f2a44; margin-top: 2px; }

/* ---------- Factor chips ---------- */
.factors-title { font-family: 'Fraunces', serif; font-size: 18px; color: #0f2a44; margin: 22px 0 10px 0; text-align: left; }
.factor {
    text-align: left; font-size: 14px; padding: 10px 12px; border-radius: 10px; margin-bottom: 8px;
}
.factor-bad  { background: #fbe4e1; color: #7d1d17; border-left: 4px solid #b3261e; }
.factor-good { background: #e2f3e7; color: #17532a; border-left: 4px solid #1e7a3c; }
.factor-none { background: #f6f3ec; color: #7a7466; }

/* ---------- Footer ---------- */
.footer { text-align: center; color: #9a9384; font-size: 13px; margin-top: 40px; }
.footer a { color: #0f2a44; font-weight: 700; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 4. Hero banner
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">Machine Learning · Hospitality</div>
    <h1>Will this guest show up?</h1>
    <p>Predict hotel booking cancellations before they happen, so the front desk can plan
       overbooking, deposits and reminder calls with confidence.</p>
    <div class="hero-stats">
        <div class="hero-stat"><b>83.2%</b><span>Test accuracy</span></div>
        <div class="hero-stat"><b>0.91</b><span>ROC-AUC</span></div>
        <div class="hero-stat"><b>79%</b><span>Cancellations caught</span></div>
        <div class="hero-stat"><b>87K</b><span>Bookings trained on</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 5. Dropdown options (from the trained encoder)
#    order = hotel, meal, country, market_segment, distribution_channel,
#            reserved_room_type, deposit_type, customer_type, agent_id
# ------------------------------------------------------------
hotel_options = list(encoder.categories_[0])
country_options = list(encoder.categories_[2])
market_options = list(encoder.categories_[3])
channel_options = list(encoder.categories_[4])
room_options = list(encoder.categories_[5])
customer_options = list(encoder.categories_[7])

# Friendly names shown to the user -> codes the model understands
meal_names = {
    "Bed & Breakfast": "BB",
    "Half Board (breakfast + dinner)": "HB",
    "Full Board (all meals)": "FB",
    "Room only (no meal)": "SC",
}

deposit_names = {
    "No deposit": "No Deposit",
    "Refundable deposit": "Refundable",
    "Non-refundable deposit": "Non Refund",
}

country_names = {
    "PRT": "Portugal", "GBR": "United Kingdom", "FRA": "France", "ESP": "Spain",
    "DEU": "Germany", "ITA": "Italy", "IRL": "Ireland", "BEL": "Belgium",
    "BRA": "Brazil", "NLD": "Netherlands", "USA": "United States", "CHE": "Switzerland",
    "CN": "China", "AUT": "Austria", "SWE": "Sweden", "CHN": "China",
    "POL": "Poland", "ISR": "Israel", "RUS": "Russia", "NOR": "Norway",
    "IND": "India", "AUS": "Australia", "JPN": "Japan", "CAN": "Canada",
    "DNK": "Denmark", "FIN": "Finland", "ROU": "Romania", "LUX": "Luxembourg",
    "ARE": "UAE", "KOR": "South Korea", "MAR": "Morocco", "TUR": "Turkey",
}

customer_names = {
    "Transient": "Individual traveller",
    "Transient-Party": "Individual (linked to a group)",
    "Contract": "Contract / allotment",
    "Group": "Group",
}


def show_country(code):
    return code + " · " + country_names.get(code, "")


def show_customer(code):
    return customer_names.get(code, code)


# ------------------------------------------------------------
# 6. Layout: form on the left, result on the right
# ------------------------------------------------------------
left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Booking details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Fill in the three tabs, then press Predict.</div>',
                unsafe_allow_html=True)

    with st.form("booking_form"):

        tab1, tab2, tab3 = st.tabs(["📅  Stay", "👨‍👩‍👧  Guest", "🧾  Booking channel"])

        # ---------------- Tab 1: Stay ----------------
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                hotel = st.selectbox("Hotel", hotel_options)
                arrival_date = st.date_input("Arrival date", datetime.date(2026, 8, 10))
                weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=20, value=1)
                reserved_room = st.selectbox("Room type reserved", room_options)
            with c2:
                lead_time = st.slider("Lead time (days booked in advance)",
                                      min_value=0, max_value=700, value=60)
                adr = st.number_input("Price per night (€)",
                                      min_value=0.0, max_value=1000.0, value=100.0, step=5.0)
                week_nights = st.number_input("Week nights", min_value=0, max_value=50, value=2)
                meal_label = st.selectbox("Meal plan", list(meal_names.keys()))

        # ---------------- Tab 2: Guest ----------------
        with tab2:
            c3, c4, c5 = st.columns(3)
            with c3:
                adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
            with c4:
                children = st.number_input("Children", min_value=0, max_value=10, value=0)
            with c5:
                babies = st.number_input("Babies", min_value=0, max_value=10, value=0)

            c6, c7 = st.columns(2)
            with c6:
                country = st.selectbox("Guest country", country_options,
                                       index=country_options.index("PRT"),
                                       format_func=show_country)
                customer_type = st.selectbox("Customer type", customer_options,
                                             index=customer_options.index("Transient"),
                                             format_func=show_customer)
                repeated_guest = st.radio("Stayed with us before?", ["No", "Yes"], horizontal=True)
            with c7:
                previous_cancellations = st.number_input("Past cancellations", min_value=0, max_value=30, value=0)
                previous_not_canceled = st.number_input("Past completed stays", min_value=0, max_value=80, value=0)
                special_requests = st.slider("Special requests (high floor, twin beds...)",
                                             min_value=0, max_value=5, value=0)

        # ---------------- Tab 3: Booking channel ----------------
        with tab3:
            c8, c9 = st.columns(2)
            with c8:
                market_segment = st.selectbox("Market segment", market_options,
                                              index=market_options.index("Online TA"),
                                              help="TA = Travel Agent, TO = Tour Operator")
                channel = st.selectbox("Distribution channel", channel_options,
                                       index=channel_options.index("TA/TO"))
                deposit_label = st.selectbox("Deposit", list(deposit_names.keys()))
                company = st.radio("Booked by a company?", ["No", "Yes"], horizontal=True)
            with c9:
                agent = st.number_input("Travel agent ID (0 = booked directly)",
                                        min_value=0, max_value=600, value=9)
                parking = st.number_input("Parking spaces needed", min_value=0, max_value=3, value=0)
                booking_changes = st.number_input("Changes made to booking", min_value=0, max_value=20, value=0)
                waiting_days = st.number_input("Days on waiting list", min_value=0, max_value=400, value=0)

        st.write("")
        submitted = st.form_submit_button("Predict cancellation risk  →", use_container_width=True)


# ------------------------------------------------------------
# 7. Convert inputs into the same format as the CSV
# ------------------------------------------------------------
meal = meal_names[meal_label]
deposit = deposit_names[deposit_label]

arrival_month = arrival_date.strftime("%B")
arrival_week = arrival_date.isocalendar()[1]
arrival_day = arrival_date.day

if agent == 0:
    agent_value = None
else:
    agent_value = agent

if company == "Yes":
    company_value = 1
else:
    company_value = None

if repeated_guest == "Yes":
    repeated_value = 1
else:
    repeated_value = 0

booking = {
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_month": arrival_month,
    "arrival_date_week_number": arrival_week,
    "arrival_date_day_of_month": arrival_day,
    "stays_in_weekend_nights": weekend_nights,
    "stays_in_week_nights": week_nights,
    "adults": adults,
    "children": children,
    "babies": babies,
    "meal": meal,
    "country": country,
    "market_segment": market_segment,
    "distribution_channel": channel,
    "is_repeated_guest": repeated_value,
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled": previous_not_canceled,
    "reserved_room_type": reserved_room,
    "booking_changes": booking_changes,
    "deposit_type": deposit,
    "agent": agent_value,
    "company": company_value,
    "days_in_waiting_list": waiting_days,
    "customer_type": customer_type,
    "adr": adr,
    "required_car_parking_spaces": parking,
    "total_of_special_requests": special_requests,
}

total_nights = weekend_nights + week_nights
total_guests = adults + children + babies
booking_value = adr * total_nights


# ------------------------------------------------------------
# 8. Result panel
# ------------------------------------------------------------
with right:
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cancellation risk for this booking.</div>',
                unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div class="result-card empty-card">
            <div class="icon">🛎️</div>
            <b>No prediction yet</b><br>
            Enter the booking details and press <b>Predict</b> to see the risk score.
        </div>
        """, unsafe_allow_html=True)

    elif total_guests == 0:
        st.error("A booking needs at least one guest.")

    elif total_nights == 0:
        st.error("A stay needs at least one night.")

    else:
        prediction, probability = predict_booking(booking)
        percent = round(probability * 100, 1)
        value_at_risk = round(booking_value * probability)

        # Risk level and colour
        if prediction == 1:
            ring_color = "#b3261e"
            verdict_class = "verdict-high"
            verdict_text = "High risk · Likely to cancel"
            verdict_note = "Consider a deposit request or a reminder call."
        elif probability >= 0.30:
            ring_color = "#d18b00"
            verdict_class = "verdict-mid"
            verdict_text = "Medium risk · Probably shows up"
            verdict_note = "Worth a confirmation email closer to arrival."
        else:
            ring_color = "#1e7a3c"
            verdict_class = "verdict-low"
            verdict_text = "Low risk · Likely to show up"
            verdict_note = "This guest looks committed to the stay."

        filled = round(probability * 360)

        result_html = f"""
        <div class="result-card">
            <div class="gauge" style="background: conic-gradient({ring_color} {filled}deg, #ece7dc {filled}deg);">
                <div class="gauge-inner">
                    <b>{percent}%</b>
                    <span>Cancel risk</span>
                </div>
            </div>
            <div class="verdict {verdict_class}">{verdict_text}</div>
            <div class="verdict-note">{verdict_note}</div>
            <div class="mini-grid">
                <div class="mini"><span>Stay</span><b>{total_nights} nights</b></div>
                <div class="mini"><span>Guests</span><b>{total_guests}</b></div>
                <div class="mini"><span>Booking value</span><b>€{round(booking_value):,}</b></div>
                <div class="mini"><span>Value at risk</span><b>€{value_at_risk:,}</b></div>
            </div>
        """

        # Key factors (based on patterns found in the data)
        factors_html = '<div class="factors-title">What drives this score</div>'
        factor_count = 0

        if deposit == "Non Refund":
            factors_html += '<div class="factor factor-bad">Non-refundable deposit — 99% of these were cancelled in the data</div>'
            factor_count += 1
        if lead_time > 180:
            factors_html += '<div class="factor factor-bad">Booked 6+ months ahead — plans change over long lead times</div>'
            factor_count += 1
        if previous_cancellations > 0:
            factors_html += '<div class="factor factor-bad">Has cancelled before — 92% of such guests cancelled again</div>'
            factor_count += 1
        if market_segment == "Groups":
            factors_html += '<div class="factor factor-bad">Group booking — groups cancel about 61% of the time</div>'
            factor_count += 1
        if country == "PRT":
            factors_html += '<div class="factor factor-bad">Home-market guest (Portugal) — 57% cancelled vs ~20% for foreign guests</div>'
            factor_count += 1
        if special_requests == 0:
            factors_html += '<div class="factor factor-bad">No special requests — these guests cancel about twice as often</div>'
            factor_count += 1
        if special_requests > 0:
            factors_html += '<div class="factor factor-good">Made special requests — a sign of a committed guest</div>'
            factor_count += 1
        if parking > 0:
            factors_html += '<div class="factor factor-good">Parking requested — almost none of these cancelled</div>'
            factor_count += 1
        if lead_time <= 7:
            factors_html += '<div class="factor factor-good">Last-minute booking — very few of these cancel</div>'
            factor_count += 1
        if repeated_value == 1:
            factors_html += '<div class="factor factor-good">Returning guest — loyal guests cancel less</div>'
            factor_count += 1

        if factor_count == 0:
            factors_html += '<div class="factor factor-none">No single strong signal — the score comes from the overall booking pattern.</div>'

        st.markdown(result_html + factors_html + "</div>", unsafe_allow_html=True)

        st.caption(f"A booking is flagged as 'likely to cancel' when risk is {round(threshold * 100, 1)}% or higher "
                   f"(threshold tuned for the best F1 score).")


# ------------------------------------------------------------
# 9. About the model
# ------------------------------------------------------------
st.write("")
with st.expander("ℹ️  About this model"):
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
**Model:** XGBoost, tuned with GridSearchCV
(compared with Logistic Regression and Random Forest)

**Data:** Hotel Booking Demand dataset — one city hotel and one resort hotel
in Portugal, 2015–2017. After removing duplicates: ~87,000 bookings.

**Class imbalance:** handled with `scale_pos_weight` and a tuned decision threshold.
""")
    with a2:
        st.markdown("""
| Metric (test set) | Score |
|---|---|
| Accuracy | 83.2% |
| ROC-AUC | 0.91 |
| Recall (cancellations caught) | 79% |
| F1 score | 0.72 |

Leakage columns (reservation status, assigned room) were removed,
so these scores reflect real-world performance.
""")


# ------------------------------------------------------------
# 10. Footer
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
    Built by <b>Rajat Rana</b> ·
    <a href="https://github.com/ranarajat994" target="_blank">GitHub</a> ·
    <a href="https://linkedin.com/in/rajat-rana-86365991" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
