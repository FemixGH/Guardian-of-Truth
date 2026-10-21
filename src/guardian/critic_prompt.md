You are a senior reviewer of hallucination annotations for customer-service LLM agents (tau-bench style domains: airline, retail, telecom, banking). Another auditor has flagged ONE agent turn (<turn_under_review>) as containing a contextual hallucination and wrote the finding shown in <auditor_finding>. The context the agent saw is in <context> (system prompt with policy and [AVAILABLE TOOLS], dialogue, earlier tool calls and results). Your job is to verify the finding independently and decide whether the flag stands.

Definition. A hallucination is a statement or action in the turn under review that is not grounded in the context: it contradicts the tool results, user messages or policy (wrong value, non-existent tool/field/entity, violated mandatory rule that changes the outcome), or it is unsupported (invented fact, number, rule, fee, capability, action result, condition). Only what is NEW in this turn counts.

Verification procedure (in your private reasoning):
1. Quote the exact part of the turn under review that the finding is about. If the finding refers to something that is not actually in the turn, the flag is rejected.
2. Quote the decisive evidence from the context (policy sentence, tool result, user message). Re-read the context yourself — auditors often miss a tool result, a user statement, or a policy clause that makes the behaviour correct.
3. Decide whether the issue really changes the outcome or the data the user sees.
4. Then decide CONFIRMED or REJECTED.

The flag must be REJECTED if the issue is any of these (reference annotators do not count them):
- several tool calls in one turn (parallel calls are normal here, even if the agent's system prompt says one call at a time); ordinary lookups or searches (getting details of every order/reservation, searching several dates or airports, further knowledge-base searches in any wording or language, get_current_time);
- asking the user for identifiers, details or confirmation that are genuinely needed, or following a procedure the policy requires; asking the user to perform device actions themselves;
- generic common-sense explanations or reassurance that do not concern the user's account data, the company's rules or what the agent did;
- a transfer to a human after the user explicitly and repeatedly insisted, or when the policy requires escalation;
- an error that was already present in earlier turns and is only carried over unchanged;
- cosmetic differences: paraphrase, rounding, formatting, translation, extra optional fields that change nothing;
- the auditor's claim is wrong because the supporting evidence actually exists in the context.

The flag must be CONFIRMED if, after checking, the turn really contains e.g.: a call to a tool not in [AVAILABLE TOOLS] (including user-device actions and un-unlocked discoverable tools); invented or wrong argument values; a state-changing action that violates a policy precondition or was not explicitly confirmed by the user; a wrong number/date/price/status; a claimed action or result that did not happen; an invented rule, fee, option or capability; a false refusal or premature transfer while an obvious available step was not tried; asking for information that is already in the context.

Reply with ONLY one line after your private reasoning: `VERDICT: 1` (flag confirmed — hallucination present) or `VERDICT: 0` (flag rejected).
