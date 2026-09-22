# MLOps report template

Converted from `mlops_report_template.docx` (course template, Atenea).
The LaTeX version we write in lives in `reports/latex/`.

<Cover page>

<Index>

## 1 Introduction

### 1.1 Goal of the project

Brief description of what the project is about and what the problem is to be solved. It must include the success criteria of the project.

### 1.2 Teammates’ evaluation

Option 1: A complete team consensus on the following statement:

**- [ ] “All team members agree that they had an equal contribution to this delivery and project: completing a fair share of the team's work with acceptable/high quality, keeping commitments, and completing assignments on time, helping teammates who are having difficulty when it is easy or important”.**

Option 2: There is no consensus on the above statement. Each team member evaluates (see scale 1 to 5 below) his/her peers’ contributions in each delivery according to their contribution, based on CATME ratings.

Figure 1. CATME ratings (from [https://info.catme.org/features/catme-five-dimensions/](https://info.catme.org/features/catme-five-dimensions/) )

|   Rating | Description of Rating                                                                                                                                                                             |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        5 | Does more or higher-quality work than expected.  Makes important contributions that improve the team's work.  Helps teammates who are having difficulty completing their work.                    |
|        4 | Demonstrates behaviors described immediately above and below.                                                                                                                                     |
|        3 | Completes a fair share of the team's work with acceptable quality.  Keeps commitments and completes assignments on time.  Helps teammates who are having difficulty when it is easy or important. |
|        2 | Demonstrates behaviors described immediately above and below.                                                                                                                                     |
|        1 | Does not do a fair share of the team's work. Delivers sloppy or incomplete work.                                                                                                                  |

**Fill in the following table (received evaluations in rows):**

|        | Name 1   | Name 2   | Name 3   | Name 4   | Name 5   | Average   |
|--------|----------|----------|----------|----------|----------|-----------|
| Name 1 | -        |          |          |          |          |           |
| Name 2 |          | -        |          |          |          |           |
| Name 3 |          |          | -        |          |          |           |
| Name 4 |          |          |          | -        |          |           |
| Name 5 |          |          |          |          | -        |           |

## 2 Methodology

### 2.1 Milestone 1: Inception

#### 2.1.1 Selection of problem and requirements engineering for ML

How, and why has the problem been selected, and what are the system requirements?

#### 2.1.2 Dataset card

Brief description of the main points of the dataset card, and link to the file in the repository.

#### 2.1.3 Model card

Brief description of the main points of the model card, and link to the file in the repository.

#### 2.1.4 Project coordination and communication

What tools have been selected, why, and how are they used?

#### 2.1.5 Selection of the cloud provider

Justification of the selection (to be used in Milestone 4)

### 2.2 Milestone 2: Model Building – Reproducibility

Describe how the different software engineering practices and tools have been applied.

#### 2.2.1 Project structure

(e.g., Cookiecutter)

#### 2.2.2 Code versioning

(e.g., GitHub Flow)

#### 2.2.3 Data versioning

(e.g., DVC)

#### 2.2.4 Experiment tracking

(e.g., MLflow)

### 2.3 Milestone 3: Model Building – Quality Assurance

Describe how the different software engineering practices and tools have been applied.

#### 2.3.1 Energy efficiency awareness

(e.g., Codecarbon) Reason about the energy efficiency of your ML component.

#### 2.3.2 Static code analysis

(e.g., pylint)

#### 2.3.3 Model testing

(e.g., pytest)

#### 2.3.4 Data testing

(e.g., Great Expectation, Deepchecks)

### 2.4 Milestone 4: Model deployment – API

Describe how you have deployed your model and the technologies used.

#### 2.4.1 ML-based component/system architecture

Description of the physical and logical architectures of the system, including diagrams. Design patterns used. Selection and justification of design decisions (e.g., cloud provider)

#### 2.4.2 API design

(e.g., FastAPI)

#### 2.4.3 API testing

(e.g., FastAPI, pytest)

### 2.5 Milestone 5: Model deployment – Model packaging

Describe how the different software engineering practices and tools have been applied.

#### 2.5.1 Container and Orchestration

(e.g., Docker, Docker compose)

#### 2.5.2 CI/CD

(e.g., GitHub Actions)

### 2.6 Milestone 5: Monitoring

Describe how the different software engineering practices and tools have been applied.

#### 2.6.1 Resource monitoring

(e.g., Grafana, Prometheus)

#### 2.6.2 Model performance

(e.g., Alibi detect)

#### 2.6.3 Cycle: Feedback loops and retraining

Description of all the stages of the system pipeline, and how they are connected. Explain how the pipeline can be used to automate the retraining process.

## 3 Self-evaluation of the project (retrospective)

To do in the final deliverable

Explain the main challenges, barriers, and opportunities you encountered during the project. Describe what you have learned, what you would do differently, etc.

Explain what concepts you have used from other courses.
