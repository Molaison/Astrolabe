------------------------- MODULE RuntimeRecovery -------------------------
EXTENDS Naturals, FiniteSets
CONSTANTS Slots, Budget, ReplayUnknown, RefundUnknown, PersistIssue
VARIABLES status, calls, pc, slot
vars == <<status, calls, pc, slot>>

Init ==
    /\ status = [s \in Slots |-> "absent"]
    /\ calls = [s \in Slots |-> 0]
    /\ pc = "idle"
    /\ slot \in Slots

Used == Cardinality({s \in Slots :
    status[s] # "absent" /\ (~RefundUnknown \/ status[s] # "unknown")})

\* Runtime.call: one committed reserved row consumes one budget unit.
Reserve(s) ==
    /\ pc = "idle" /\ status[s] = "absent" /\ Used < Budget
    /\ status' = [status EXCEPT ![s] = "reserved"]
    /\ pc' = "reserved" /\ slot' = s
    /\ UNCHANGED calls

ResumeReserved(s) ==
    /\ pc = "idle" /\ status[s] = "reserved"
    /\ pc' = "reserved" /\ slot' = s
    /\ UNCHANGED <<status, calls>>

\* The issued SQLite transaction commits before operation.invoke begins.
Issue ==
    /\ pc = "reserved"
    /\ status' = IF PersistIssue THEN [status EXCEPT ![slot] = "issued"] ELSE status
    /\ pc' = "issued"
    /\ UNCHANGED <<calls, slot>>

Invoke ==
    /\ pc = "issued"
    /\ calls' = [calls EXCEPT ![slot] = @ + 1]
    /\ pc' = "running"
    /\ UNCHANGED <<status, slot>>

\* External work may finish before its completion is recorded.
Return ==
    /\ pc = "running"
    /\ pc' = "returned"
    /\ UNCHANGED <<status, calls, slot>>

Complete ==
    /\ pc = "returned"
    /\ status' = [status EXCEPT ![slot] = "completed"]
    /\ pc' = "idle"
    /\ UNCHANGED <<calls, slot>>

Fail(outcome) ==
    /\ pc = "running"
    /\ status' = [status EXCEPT ![slot] = outcome]
    /\ pc' = "idle"
    /\ UNCHANGED <<calls, slot>>

\* Process death loses control state, never a committed journal row.
Crash ==
    /\ pc # "idle"
    /\ pc' = "idle"
    /\ UNCHANGED <<status, calls, slot>>

ObserveIssued(s) ==
    /\ pc = "idle" /\ status[s] = "issued"
    /\ status' = [status EXCEPT ![s] = "unknown"]
    /\ UNCHANGED <<pc, calls, slot>>

\* This action exists only in the deliberately broken model.
Replay(s) ==
    /\ ReplayUnknown /\ pc = "idle" /\ status[s] = "unknown"
    /\ pc' = "reserved" /\ slot' = s
    /\ UNCHANGED <<status, calls>>

Next ==
    \/ \E s \in Slots : Reserve(s) \/ ResumeReserved(s)
                        \/ ObserveIssued(s) \/ Replay(s)
    \/ Issue \/ Invoke \/ Return \/ Complete \/ Crash
    \/ Fail("unknown") \/ Fail("failed")

Spec == Init /\ [][Next]_vars
TypeOK ==
    /\ status \in [Slots -> {"absent", "reserved", "issued",
                             "unknown", "completed", "failed"}]
    /\ calls \in [Slots -> Nat]
    /\ pc \in {"idle", "reserved", "issued", "running", "returned"}
    /\ slot \in Slots
AtMostOnce == \A s \in Slots : calls[s] <= 1
BudgetNeverRefunded ==
    Cardinality({s \in Slots : status[s] # "absent"}) <= Budget
IssuedBeforeInvoke ==
    \A s \in Slots : calls[s] > 0 =>
        status[s] \in {"issued", "unknown", "completed", "failed"}
=============================================================================
