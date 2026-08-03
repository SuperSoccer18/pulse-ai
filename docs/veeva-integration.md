# Veeva Integration

`veeva_client.py` owns all live Veeva API access used by `server.py`. Auth
and SOQL/PATCH patterns mirror what was validated (and can be re-explored)
in `veeva_integration_test.py`.

## Environment status: **sandbox only, no production plan yet**

This entire integration targets a Veeva **sandbox/dev org**. There is
currently no plan for a production Veeva integration — treat everything
below as demo scaffolding, not a foundation to build production
multi-rep support directly on top of. Expect a real integration effort to
revisit rep identity, product sourcing, and token handling from scratch
rather than incrementally generalizing the current code.

## Hardcoded demo rep

`ARTHUR_ID = "005G0000001AxUMIA0"` (Arthur Stephenson) is hardcoded in both
`veeva_client.py` and `testAgents.py`/`veeva_integration_test.py`. There is
no rep login, session, or multi-rep support anywhere in `server.py` — every
`/api/calls` request and every submit is scoped to this one demo rep. This
is the first thing to rip out and replace for anything beyond a demo.

## Auth

`get_veeva_token()` does a client-credentials OAuth POST to `LOGIN_URL`
(`VEEVA_CLIENT_ID`/`VEEVA_CLIENT_SECRET`) on **every call** — no token
caching. Explicitly called out as fine for demo request volume, but would
need caching (and probably retry/refresh handling) before any real traffic.

## Reads — `fetch_calls_for_rep()`

SOQL query against `Call2_vod__c`, filtered to the demo rep, excluding
`Cancelled_vod`, ordered by `Call_Date_vod__c`. Mapped to the frontend's
expected shape: `id`, `hcp_name`, `specialty`, `location`, `product`,
`status`.

Two behaviors worth knowing before changing this function:

1. **Locked-call re-recording produces duplicate HCP rows.** Once
   submitted, Veeva locks a call record — `Status_vod__c` can never move off
   `Submitted_vod` again (confirmed live: a 400
   `FIELD_CUSTOM_VALIDATION_EXCEPTION` when attempting to reset one for
   re-testing). Re-recording for the same HCP means creating a brand new
   `Call2_vod__c` row rather than reusing the old one — which can leave one
   HCP with both a `Submitted_vod` (done) row and a fresh `Planned_vod`
   (pending) row simultaneously.
2. **One row per HCP, done rows still queried (not just excluded).**
   `fetch_calls_for_rep()` keeps querying `Submitted_vod` calls (rather than
   filtering them out entirely) so the "Documented" stat and done-card
   styling on the call list stay accurate on a fresh page load — not just
   right after a submit in the same browser session. But per HCP, only one
   row is surfaced: an actionable **pending** call is preferred over a
   **done** one, since a done call has nothing left for the rep to do. See
   the `by_account` merge logic for the exact tie-breaking rule.

## Writes — `update_call_in_veeva()`

A PATCH against `Call2_vod__c/{call_id}`. `fields` is open-ended at this
function's level — the caller (`server.py`'s `_extraction_to_veeva_fields()`)
is responsible for only including confirmed-writable fields.

### Confirmed field-writability limits

**Product/indication/next-steps/competitor data has no writable home.**
Neither `Call2_vod__c` nor `Call2_Detail_vod__c` (the latter is
read/write-locked for our integration user's permission set) can hold this
data — it stays **UI-only, never persisted to Veeva**. This was determined
by exploration in `veeva_integration_test.py` (its `describe()` dump against
`Call2_Detail_vod__c` prints create/update permission per field — rerun that
if permission sets ever change). Confirm this is still true before assuming
otherwise; permission sets can change without this repo's knowledge.

### Extraction → Veeva field mapping (`server.py::_extraction_to_veeva_fields`)

| Extraction field | Veeva field | Notes |
|---|---|---|
| `call_metadata.location` | `Territory_vod__c` | |
| `call_metadata.call_datetime` (date part) | `Call_Date_vod__c` | first 10 chars, always set if present |
| `call_metadata.call_datetime` (if it has a time component) | `Call_Datetime_vod__c` | see timezone caveat below |
| `call_metadata.engagement_method` | `Call_Channel_vod__c` | |
| `call_metadata.virtual_engagement_tool` | `Remote_Meeting_Type_vod__c` | |
| `summary` | `Chat_Summary_vod__c` | |
| — | `Status_vod__c` | always hardcoded to `"Submitted_vod"` on submit |

**Unverified assumption — datetime timezone.** `field-extractor` can return
either a full ISO datetime (`"2026-07-15T10:00:00"`, no offset) or a
date-only ISO string. When there's a time component, the code appends `"Z"`
if no timezone marker is already present. This is flagged in the code
itself as `NOTE (unverified)`: Salesforce datetime fields typically expect
a timezone-qualified ISO string, and our extraction produces a naive local
value with no real offset — appending `"Z"` is a best-effort guess, not a
confirmed-correct format. **Watch the first live submit for a 400
specifically on `Call_Datetime_vod__c`** if this hasn't been exercised
against real Veeva data yet.

**Placeholder product data.** `_PRODUCT_PLACEHOLDER` in `veeva_client.py`
hardcodes a product per demo HCP account, mirroring the products seeded
onto 6 dummy calls created via `veeva_integration_test.py`'s `DUMMY_CALLS`.
This is stand-in data only, matching the fact that real product data
(`Call2_Detail_vod__c`) is locked — see above. Do not treat this dict as
anything but demo scaffolding.

**Status mapping** (`_STATUS_MAP`): Veeva's `Status_vod__c` picklist
collapsed to the UI's pending/done binary — `Submitted_vod → done`,
`Saved_vod`/`Planned_vod → pending`. `Cancelled_vod` is excluded via the
SOQL `WHERE` clause, not this map.

## `veeva_integration_test.py`

Exploration/setup script, not an automated test suite despite the name —
it's how the 6 dummy calls (`DUMMY_CALLS`) and their linked
`Call2_Detail_vod__c` product records were originally created in the
sandbox, and how field-level create/update permissions were discovered
(`describe()` dump, printed as a table). Re-run pieces of this if you need
to re-seed demo data or re-check permission sets after a sandbox refresh.
`EXISTING_CALL_IDS` documents the call IDs created in the previous run,
keyed by HCP account ID — useful if you need to clean up or reference
those specific records.
