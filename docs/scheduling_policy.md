# Scheduling policy (ClinicAI)

This document describes the unified slot-selection policy used by Telegram FSM booking, `crud.find_available_slots`, and `scheduler.plan_appointment`.

## Source of truth

- **Specialty:** `Doctor.specialty` (FK on `slots.doctor_id`) is authoritative; `Slot.specialty` is kept for reporting compatibility.
- **Policy module:** `scheduler/slot_policy.py`
- **Feature flag:** `USE_SLOT_POLICY=true` in `.env` (set `false` to fall back to legacy SQL-only ordering).

## Block rules (reserved capacity)

| Patient priority | May book slot tiers |
|------------------|---------------------|
| P1 (urgent)      | P1, P2, P3, open    |
| P2 (medium)      | P2, P3, open        |
| P3 (routine)     | P3, open            |

P3 patients cannot take P1-only or P2-only reserved slots.

## Wave horizons

| Priority | Booking window (days ahead) |
|----------|----------------------------|
| P1       | 2                          |
| P2       | 7                          |
| P3       | 30                         |

If no slots exist inside the wave window, the patient is waitlisted (not shown distant slots).

## Ranking (P2/P3 load spreading)

After block + wave filters:

1. Preferred date match (if provided)
2. Block tier match (exact priority class preferred)
3. Lower clinic load per day (P2/P3)
4. Lower doctor load per day (P2/P3)
5. Lower utilization (P2/P3)
6. Earliest datetime (P1 always prefers earliest)

## GP fallback

If no slots exist for the requested specialty, `general_practice` slots are considered as fallback.

## Unsupported specialties (Telegram)

The following are **not** bookable via specialty keyboard (commented in `bot/keyboards.py`):

- cardiology (قلب)
- pediatrics (أطفال)
- dentistry (أسنان)
- ophthalmology (عيون)

When detected in free text, the FSM offers **general practice** fallback (`OFFER_GP_FALLBACK`).

Active keyboard specialties align with `scheduler.classifier.SPECIALTY_NAMES_AR` and seeded doctors.

## Patient booking guards

Before confirming:

- No overlapping active appointments (any specialty)
- No duplicate same-specialty same-day active appointments

Enforced in `crud.find_patient_booking_conflict` and re-checked at commit in `reserve_slot_and_create_appointment`.

## Waitlist

When no slot is available:

1. `enqueue_waitlist` computes queue position
2. `create_waitlist_appointment` persists profile + waitlisted appointment
3. User message includes: `تمت إضافتك للانتظار — موقعك: N`

## FSM session persistence

Active booking sessions are stored in `fsm_sessions` (24h TTL cleanup via `crud.cleanup_stale_fsm_sessions`).
