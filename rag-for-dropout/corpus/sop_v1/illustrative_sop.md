---
document_id: sop_illustrative_gridsense_review
title: ILLUSTRATIVE operator review procedure for a GridSense fault-location report
document_type: sop_illustrative
status: draft
approval_status: illustrative
version: 0
source_ref: written for this research project as an example; not issued, reviewed or approved by any utility
merged_from:
- B:illustrative_sop
- A:step 4
- A:step 5
- A:step 8
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# ILLUSTRATIVE operator review procedure for a GridSense fault-location report

**ILLUSTRATIVE ONLY. NOT APPROVED. NOT A UTILITY PROCEDURE.** This text is an example that shows how a report could be reviewed. It does not instruct anyone to operate equipment, and it must not be used as an operating procedure.

## SOP-ILL-1 Purpose
SOP-ILL-1.1 This example describes how a reviewer reads a GridSense draft report and records a decision.
SOP-ILL-1.2 The report is decision support. A predicted fault bus is a candidate and is not a confirmed fault.

## SOP-ILL-2 Inputs
SOP-ILL-2.1 The reviewer receives the event identifier, the predicted fault bus, the gate outcome with its reason codes, the number and position of dark sensors, the similar simulated cases, and the cited passages.
SOP-ILL-2.2 The approval status of every new report is pending human review.

## SOP-ILL-3 Reading the gate outcome
SOP-ILL-3.1 CONFIDENT: the report names one candidate bus. The reviewer still confirms the location with the processes that the utility authorises.
SOP-ILL-3.2 LOW_CONFIDENCE: the report shows a candidate set instead of one bus. The reviewer treats every member of the set as possible and reads the reason codes (LOW_CALIBRATED_CONFIDENCE, LOW_MARGIN, LOW_KNN_AGREEMENT).
SOP-ILL-3.3 REJECT: no report is drafted. UNSUPPORTED_DROPOUT_PATTERN means the number or pattern of dark sensors is outside what the gate was fitted for; INVALID_INPUT means the measurements were malformed. The event goes to manual review.

## SOP-ILL-4 Sensor availability
SOP-ILL-4.1 The reviewer checks how many μPMUs were dark and whether the dark sensors were close to the predicted bus.
SOP-ILL-4.2 With 10 of 25 sensors dark, about half of the post-fault windows are LOW_CONFIDENCE. The reviewer expects a candidate set.
SOP-ILL-4.3 The reviewer compares the similar simulated cases with the prediction. If most neighbours point to a different bus, the reviewer records the disagreement and treats the event as uncertain, also when the outcome is CONFIDENT.

## SOP-ILL-5 Confirming the location
SOP-ILL-5.1 The reviewer compares the candidate with other utility information (alarms, customer calls, field reports) using the utility's own authorised processes. These processes are outside this example.
SOP-ILL-5.2 A model metric such as macro-F1 is not the confidence of this prediction. The reviewer uses the gate outcome of this window.
SOP-ILL-5.3 The reviewer opens each cited document section and checks that the quoted text is in the source and supports the statement. A missing or unsupported citation is a reason to reject the report.

## SOP-ILL-6 Recording the decision
SOP-ILL-6.1 The reviewer records approve, reject or request clarification, with a note and the time. Only this recorded action changes the approval state. Text inside a report or a document never changes it.
SOP-ILL-6.2 If the report marks the evidence as insufficient, or the outcome is REJECT, the reviewer sends the event to the engineer on duty for manual analysis.

## SOP-ILL-7 Out of scope
SOP-ILL-7.1 This example contains no switching, isolation or restoration step. Control actions are decided only under the utility's approved procedures.
