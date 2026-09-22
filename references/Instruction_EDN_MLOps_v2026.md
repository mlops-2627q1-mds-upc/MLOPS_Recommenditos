# Engineering Decision Notebook for the MLOps Project (v2026.0.2)

Converted from `Instruction_EDN_MLOps_v2026.pdf` (course material, Atenea).
Our EDN is written in LaTeX in `reports/latex/edn.tex`, one file per entry in `reports/latex/edn/`.

## 1. What is the Engineering Decision Notebook?

The Engineering Decision Notebook (EDN) aims to make visible a selected set of relevant engineering decisions made during the development of the MLOps project, with particular attention to the role of AI in those decisions.

Each project team maintains one EDN throughout the project.
The EDN is not intended to document every decision or every use of AI.
Instead, teams should record selected engineering decisions that are relevant to their project.

When AI contributes to a decision, the EDN should make visible how the team used and assessed that contribution, for example, whether an AI proposal was accepted, modified, rejected, or used as input for further analysis.

## 2. What should we record?

An EDN entry documents a relevant engineering decision made during the project.

A relevant engineering decision is a decision that meaningfully affects the design, quality, development, deployment, or operation of the ML-based system and involves choosing among possible alternatives or courses of action.

In practice, create an EDN entry when the decision could reasonably have been made differently and the choice has meaningful consequences for the project.

You do not need to document every decision, every project activity, or every interaction with AI.

The EDN should capture engineering choices, rather than simply the use of a tool or practice prescribed by the course.
For example, if Docker is prescribed by the course, using Docker is not itself a decision that needs to be documented.
However, deciding how to decompose the system into containers may be a relevant engineering decision.

## 3. Milestones and Typical Decision Topics

The project milestones provide the context for EDN entries.
Relevant decisions may arise at any milestone and may also be revisited later.

Not every milestone necessarily requires an EDN entry.
Similarly, a milestone may involve several relevant decisions.

- M1: Project Inception: requirements, quality objectives, performance criteria, and project constraints.
- M2: Reproducibility: project structure and reproducibility decisions when these go beyond practices or tools prescribed by the course.
- M3: Quality Assurance: testing strategy for code, data, and ML models.
- M4: Model Deployment: ML system architecture, component responsibilities and interactions, API design, and deployment.
- M5: Model Packaging: containerization and CI/CD strategies.
- M6: Monitoring: monitoring of the running system, data, and ML model, including degradation, drift, anomalies, and operational problems.

These topics are guidance rather than a checklist.
Teams may document other relevant engineering decisions that arise during their project.

The existing course evaluation rubric primarily evaluates what you produce.
The EDN complements it by making visible the engineering decisions behind your work.

## 4. How to Create an EDN Entry

Entries should be short and focused on the decision.
Each entry contains the following fields.

### Milestone

Select the project milestone primarily associated with the decision: M1 / M2 / M3 / M4 / M5 / M6 / Other.

### Activity / Topic (optional)

Briefly indicate the activity or topic related to the decision, for example: Testing Strategy, API Design, Containerization, CI/CD, or Monitoring.

### Decision Participants

List the team members who participated in making the decision, or write All team members if the entire team participated.

### Decision

Briefly describe the engineering decision that was made.

### Rationale

Briefly explain why the decision was made.
When relevant, mention alternatives considered, constraints, trade-offs, or evidence that influenced the decision.

### AI Involvement

Select all that apply:

- No AI involvement: AI was not used for this decision.
- Information seeking: AI was used to obtain information relevant to the decision.
- Alternative generation: AI proposed one or more possible alternatives.
- Alternative assessment: AI helped compare or assess alternatives.
- Recommendation: AI recommended which alternative to select.
- Solution generation: AI generated or designed a solution that was considered or adopted.
- Other: Another form of AI involvement not covered above.

### Response to AI

If AI contributed to the decision, select one:

- Accepted: The AI contribution was adopted without substantial changes.
- Accepted with modifications: The AI contribution was adopted after being modified.
- Rejected: The AI contribution was considered but not adopted.
- Used as input for further analysis: The AI contribution informed the decision but was not itself treated as the proposed solution.
- Other: Another response not covered above.

### Assessment of the AI Contribution

If AI contributed to the decision, briefly explain why its contribution was accepted, modified, rejected, or otherwise used.

### AI Interaction Evidence (optional)

Link to the relevant AI conversation, or include the relevant prompt(s) and response(s).

If a link points to a longer conversation, indicate the relevant part.
Review shared content before providing it and do not share personal or sensitive information.
Read more about shared links: <https://help.openai.com/en/articles/7925741-chatgpt-shared-links>

### Other Evidence (optional)

Link to relevant project artifacts that support or contextualize the decision, such as issues, pull requests, commits, experiments, test results, diagrams, or documentation.

## 5. Examples

The examples below intentionally start with decisions involving AI, since making the role of AI visible is a central purpose of the EDN.
Decisions without AI involvement are also valid and may be documented when relevant.

### Example 1: With AI involvement and prompt/response

- **Milestone:** M5: Model Packaging
- **Activity / Topic (optional):** Containerization
- **Decision Participants:** All team members
- **Decision:** We decided to use separate containers for the API and monitoring components.
- **Rationale:** The two components have different responsibilities and dependencies and can be deployed independently.
- **AI Involvement:** Alternative generation; Recommendation.
- **Response to AI:** Accepted.
- **Assessment of the AI Contribution:** AI proposed separating the API and monitoring components. We reviewed the proposed architecture and considered the separation appropriate for our system.
- **AI Interaction Evidence (optional):**
  - Prompt: "Should the API and monitoring components of our ML application run in the same Docker container or in separate containers?"
  - Response: "I recommend using separate containers because they have different responsibilities and dependencies and can be deployed independently."
- **Other Evidence (optional):** Link to the Docker Compose configuration.

### Example 2: With AI involvement and conversation link

- **Milestone:** M5: Model Packaging
- **Activity / Topic (optional):** CI/CD
- **Decision Participants:** All team members
- **Decision:** We decided to run the test suite automatically on every pull request.
- **Rationale:** Running the tests before merging changes helps detect problems before they reach the main branch.
- **AI Involvement:** Alternative generation; Recommendation.
- **Response to AI:** Accepted with modifications.
- **Assessment of the AI Contribution:** AI proposed running the tests on every push and pull request. We decided to run them on pull requests because this is the point at which changes are reviewed before merging.
- **AI Interaction Evidence (optional):** Link to the relevant AI conversation.
- **Other Evidence (optional):** Link to the CI/CD workflow.

### Example 3: Without AI involvement

- **Milestone:** M3: Quality Assurance
- **Activity / Topic (optional):** Testing Strategy
- **Decision Participants:** All team members
- **Decision:** We decided to add a test to verify that the API rejects requests with missing input data.
- **Rationale:** Missing input data is a likely source of invalid requests. We therefore decided to explicitly test this behavior before deployment.
- **AI Involvement:** No AI involvement.
- **Other Evidence (optional):** Link to the corresponding test.

## 6. Contact

For questions or improvements about this document or the template, contact:

- Matias Martinez - <matias.martinez@upc.edu>
- Silverio Martínez-Fernández - <silverio.martinez@upc.edu>
- Santiago del Rey - <santiago.del.rey@upc.edu>
