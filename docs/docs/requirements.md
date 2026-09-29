Requirements
============

What the used-car price component must do and which qualities it must have, stated independently of how it is built.
The requirements are derived from the use cases UC1 (car valuation) and UC2 (purchase guidance) in the [project brief](project-brief.md).

This page deliberately fixes no design: no endpoints, no data formats, no libraries, no deployment mechanics.
How each requirement is realised, in the detail a developer builds and tests against, is the [specification](specification.md), which uses the same IDs.
What the model learns and how good it has to be is the [problem specification](problem-spec.md).

Every requirement has an ID (`FR-xx` for functional, `NFR-xx` for non-functional), a priority and an acceptance criterion.
The IDs are the unit of traceability: tests carry them, and CI turns them into a requirement-to-test matrix (NFR-07) that the report cites as evidence.

Priorities follow MoSCoW.
**Must** means the component is not worth delivering without it.
**Should** means it is part of the plan and is dropped only if a milestone is at risk.

Status markers as in the project brief:

- **[decided]** agreed by the team.
- **[proposed]** proposed and pending team confirmation; the alternatives are in the [specification](specification.md#decisions-pending-confirmation).
- **[open]** depends on a decision that is not made yet.

## Who the component is for

- **The seller** owns a car and wants to offer it at a realistic price without comparing listings by hand (UC1).
  They know their own car, but not the market.
- **The buyer** is looking at a car, or at a kind of car, and wants to know whether the asking price is normal (UC2).
  They can describe what they want only roughly.
- **The team** builds, releases and operates the component, and has to be able to tell at any time whether it is still right.
  Several requirements exist for this role alone; they are marked as such.
- **Listed people**, whose listings the model was trained on, are not users but are affected by the component, which is what NFR-08 protects.

## 1. Functional requirements

Which properties of a car can be described at all is the feature set of the [problem specification](problem-spec.md#4-features).
The requirements below say which of them a user must supply, what they get back, and what has to be true about the answer.

### What the component accepts

| ID | Priority | Requirement | Acceptance criterion |
|----|----------|-------------|----------------------|
| FR-01 | Must | **Description needed for a valuation (UC1) [decided, EDN-15].** A valuation requires a small, fixed set of facts that the owner of a car knows without research: which car it is, how old it is, how far it has run, how it is powered and driven, where it is offered and what kind of seller offers it. Every further property is optional. Leaving an optional property out is allowed and must not cost more accuracy than SC-06 of the [problem specification](problem-spec.md#8-success-criteria) permits. | A valuation that omits a required fact is refused and says which one; a valuation that omits only optional facts is answered and stays within SC-06. |
| FR-02 | Must | **Description needed for a price range (UC2).** A price range is produced from any partial description, down to the make alone. | A request naming only the make is answered; adding further properties narrows the range. |
| FR-03 | Must | **Plausibility of the input.** Every submitted fact is checked before it reaches the model: values must be of the right kind, dates and quantities must lie in ranges a real car can have, and categories must be ones the deployed model actually knows. Implausible input is refused, with all problems reported at once, each naming the fact it concerns. | Each rule is checked by its own test; one refusal reports several faults together. |
| FR-04 | Must | **Scope check.** A car whose make lies outside the supported scope of the [problem specification](problem-spec.md#2-scope) is refused rather than estimated, and the refusal states what is supported. | A request for an unsupported make is refused and the answer names the supported makes. |
| FR-05 | Must | **Unfamiliar detail in an in-scope car [decided, EDN-18].** A car that is in scope as a whole but carries a detail the deployed model has not seen, such as a market or a model name absent from its training data, is still answered, and the answer names the unfamiliar detail. | A request from the held-out market and a request with an unseen model name are both answered and both carry the warning. |

### What the component answers

| ID | Priority | Requirement | Acceptance criterion |
|----|----------|-------------|----------------------|
| FR-06 | Must | **Valuation (UC1) [decided, EDN-16].** For a described car, the component returns one price estimate in euros, which model version produced it, and a reference that identifies this individual answer later. | The answer carries an amount, a model version and a reference. |
| FR-07 | Must | **Price range (UC2) [decided, EDN-16].** For a partially described car, the component returns a typical price together with a lower and an upper bound at a stated confidence level. The typical price always lies inside the bounds. | Lower bound ≤ typical price ≤ upper bound in every answer; the coverage of the bounds is part of NFR-01. |
| FR-08 | Should | **Explanation of a valuation [decided, EDN-11].** A valuation is accompanied by the few properties that influenced this particular car's price most, each with the direction and the size of its effect, plus the combined effect of everything else. The effects are complete: together they account for the whole difference between the typical price and the estimate, so the estimate can be checked rather than only believed. | Recomputing the estimate from the reported effects reproduces it. |
| FR-09 | Should | **Comparable listings.** Alongside a price range, the component returns real listings of comparable cars with their prices, so that the range can be checked against the market. Comparability is defined by stated limits on age and mileage, and those limits are never relaxed to fill the list: when nothing is comparable, nothing is returned. The listings carry nothing personal (NFR-08). | Every returned listing satisfies the stated limits; a request with no comparable car returns an empty list. |

### What the component has to support over time

These requirements exist for the team as operator, not for the seller or the buyer.

| ID | Priority | Requirement | Acceptance criterion |
|----|----------|-------------|----------------------|
| FR-10 | Must | **Observed prices [decided, EDN-13].** The component accepts a later report of what a car was really priced at, linked to the answer it belongs to, so that the real error and the actual coverage of the price ranges can be measured instead of inferred from input statistics. Only the operators can report prices back; nothing that writes data is reachable by the public (NFR-09). | A reported price is joined to its answer; an unauthorised report is refused. |
| FR-11 | Must | **Operational visibility.** The running component reports whether it is healthy, which model version it serves, and how much traffic, latency and how many errors it sees. | The health and the traffic figures can be read from outside the component. |
| FR-12 | Must | **The model version in service [decided, EDN-08].** At any time it is unambiguous and verifiable which model version is serving, a released version is fixed and reproducible, and putting a different version into service changes nothing about what the component accepts and answers. | The version the running component reports equals the version the repository declares as released. |
| FR-13 | Must | **Record of every answer.** Every answer is recorded with what was asked, what was answered, which warnings it carried, which model version produced it and when, free of personal data (NFR-08), so that any answer can be reconstructed and analysed afterwards. | Each answer produces exactly one record containing those parts and nothing personal. |
| FR-14 | Must | **Monitoring of drift and of error.** Incoming traffic is compared with the data the model was trained on, window by window, and reported: which properties changed, how large the real error is, and whether the price ranges still hold their stated confidence level. | A monitoring run reports all three: changed properties, error and coverage. |
| FR-15 | Must | **Retraining and release [decided, EDN-12].** Retraining is started by a person reacting to a monitoring finding, never automatically. A retrained model reaches users only after it has demonstrably met every success criterion and a person has approved it. A release can be undone. | The full chain runs once end to end: finding, retraining, quality gate, approval, release, and a rehearsed rollback. |
| FR-16 | Should | **Documented contract and uniform failures.** Every operation is documented with an example of what it is asked and what it answers, and every failure is reported in one uniform shape, whatever caused it. | The documentation contains an example per operation; every failure path produces the same shape. |

### Out of scope

- **Pricing many cars at once.** The component answers one car at a time, the way a person asks about their car.
- **Describing a car in free text.** The user supplies the properties; interpreting "well kept BMW 3 series from 2018" with a language model stays an optional add-on the component never depends on ([problem specification](problem-spec.md#2-scope)).
- **Accounts and personal history.** Nobody signs in and nothing is kept per person; the same question always gets the same answer.
- **Cars outside the scope.** New cars, transporters, unsupported makes and markets outside the data are refused rather than estimated (FR-04).
- **A guaranteed sale price.** The component estimates what a car is offered for, not what it finally sells for ([problem specification](problem-spec.md#1-problem-statement)).

## 2. Non-functional requirements

The component runs on the single small machine the course provides (4 GB RAM, 20 GB disk, CPU only, **[decided]**, [EDN-17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
That machine is a given constraint, not a design choice, and NFR-02 to NFR-04 are written for it.
Those three targets are themselves **[proposed]**: they are estimates, checked against the first load test in M4 and adjusted there if needed.

| ID | Priority | Quality | Requirement | Acceptance criterion |
|----|----------|---------|-------------|----------------------|
| NFR-01 | Must | Model quality | Only a model that meets every success criterion SC-01 to SC-06 of the [problem specification](problem-spec.md#8-success-criteria) is released. This requirement sets no thresholds of its own. | The quality gate in front of a release evaluates all six criteria and blocks on any miss. |
| NFR-02 | Must | Latency | An answer arrives while the user is still looking at the screen, so that trying a variant of the same car costs nothing: 95 % of valuations within 200 ms and 95 % of price ranges and comparable listings within 300 ms. **[proposed]** | Measured under the load of NFR-03 on the target machine. |
| NFR-03 | Must | Throughput and scalability | The component serves 20 requests per second for five minutes with fewer than 1 % errors while still meeting NFR-02. Its correctness must not depend on how many copies of it are running: whatever the number, every answer appears exactly once in the record of FR-13, and every reported price of FR-10 is visible to the monitoring of FR-14. **[proposed]** | A load test at that rate, repeated with more than one copy running, with the record checked for duplicates and gaps. |
| NFR-04 | Must | Resources | The entire system fits the provided machine with headroom left at all times, and nothing in the serving path needs hardware beyond it, in particular no GPU. Storage that grows (records, metrics) is bounded, so that it cannot fill the disk. **[proposed]** | Memory and disk stay within the machine's limits during the NFR-03 load test, with the stated headroom free. |
| NFR-05 | Must | Availability | **[decided, EDN-10]** The component recovers by itself within 30 seconds of a crash or a restart of the machine, without anyone logging in. It has zero unplanned downtime during the graded presentation and its warm-up. Outside that window it runs unattended on a single machine without redundancy, so uptime is monitored and reported, not contractually promised. | A rehearsed crash and a rehearsed reboot, each followed by automatic recovery within 30 seconds; a health check immediately before the presentation. |
| NFR-06 | Must | Reproducibility | From a clean clone of the repository, the documented steps reproduce the same splits and the same metrics within ±0.1 percentage points, and every training run records which code, which data version and which parameters produced it. | A rerun from a clean clone before each delivery. |
| NFR-07 | Must | Maintainability | The repository stays reviewable for a semester with four part-time people: code and notebooks pass the agreed quality checks without findings, the test coverage of the project code is at least 80 % with every exclusion justified, and the checks run before a change is merged rather than after. Every requirement is traceable to the test that verifies it, or is explicitly marked as verified by hand. | The checks and the requirement-to-test matrix run in CI and are reviewed before each delivery. |
| NFR-08 | Must | Privacy | Nothing that identifies a listed person leaves the raw data: no such property reaches the processed data, the model, the answers, the comparable listings or the records, and the records do not identify the caller either. The raw data is not re-hosted by us. **[decided, EDN-07]** | Automated checks on the processed data, on the answers and records, and on the model's inputs. |
| NFR-09 | Must | Security | Everything reachable from the internet is read-only and carries no secret, and anything that writes data is not reachable from the internet at all. Requests are bounded in size. No secret is committed to the repository. A dependency with a known and fixable vulnerability does not reach the running system. **[decided, EDN-13]** | A check from outside the machine that the writing operation is refused; size limit test; secret and vulnerability scans in CI. |
| NFR-10 | Should | Energy efficiency | The energy cost of the component is known and small: every training run records its emissions, training the chosen configuration takes at most 15 minutes on a laptop CPU, and serving is reported as an average energy cost per answer. | Emissions recorded per training run; an energy-per-answer figure from the NFR-03 load test. |
| NFR-11 | Must | Observability | **[decided, EDN-14]** The monitoring detects the new-market scenario of the [project brief](project-brief.md) within the first 100 answers and names at least three changed properties besides the market itself, while raising at most 1 false alarm in 20 comparable windows of normal traffic. | The replay of the held-out market, plus a control run of 20 normal windows. |
| NFR-12 | Must | Portability | The whole system starts on a fresh machine with a single documented command and needs no paid or external service at runtime. | A deployment from scratch on a clean machine. |
| NFR-13 | Must | Delivery | A change merged into the main line reaches the running system by itself, is verified there, and is put back to the previous version automatically if that verification fails. Changes that affect only documentation need not be deployed. | A merge that deploys and verifies itself, and a rehearsed rollback. |

### Quality model

The `Quality` column maps to [ISO/IEC 25010](https://www.iso.org/standard/78176.html) (software product quality) and [ISO/IEC 25059](https://www.iso.org/standard/80655.html) (its extension for AI systems), except where noted.
Three rows have no equivalent in either standard; they are here because the course grades them as their own practice, not because a quality model names them.

| Quality (this page) | ISO/IEC 25010 / 25059 characteristic |
|----------------------|----------------------------------------|
| Model quality (NFR-01) | AI-specific functional correctness (25059); 25010 covers software only, not model behaviour |
| Latency, Throughput and scalability, Resources (NFR-02 to NFR-04) | Performance efficiency (Time behaviour, Capacity, Resource utilization) |
| Availability (NFR-05) | Reliability (Availability) |
| Reproducibility (NFR-06) | Reliability (Recoverability), extended: not itself a named 25010 sub-characteristic, but standard in ML-specific quality work |
| Maintainability (NFR-07) | Maintainability |
| Privacy (NFR-08) | Security (Confidentiality) |
| Security (NFR-09) | Security |
| Energy efficiency (NFR-10) | Not in 25010/25059; graded as its own M3 practice |
| Observability (NFR-11) | Maintainability (Analysability), extended for MLOps monitoring |
| Portability (NFR-12) | Portability |
| Delivery (NFR-13) | Not in 25010/25059; graded as its own M5 practice |

## Open points

The requirements above are agreed, except where a **[proposed]** marker says otherwise.
Open is how some of them are realised: the performance and resource targets of NFR-02 to NFR-04, and everything that depends on how the course machine can be reached from outside, which affects NFR-05, NFR-09 and NFR-13.
Both are listed with their alternatives under [Decisions pending confirmation](specification.md#decisions-pending-confirmation) in the specification.
