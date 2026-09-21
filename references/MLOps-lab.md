## Laboratory. Third semester Machine Learning Systems in Production (MLOps)

Instructors: Silverio Martínez-Fernández, Santiago del Rey, Matias Martinez

## Short introduction - Learning Objectives

1. Interpret the basic concepts of Software Engineering for ML systems, especially in relation to the use and exploitation of MLOps practices.
2. Apply and analyze MLOps practices to build ML models, fostering reproducibility and quality assurance.
3. Apply and analyze MLOps practices to deploy ML models, fostering API development and component delivery.
4. Describe concepts and methods related to monitoring data obtained during the use of ML systems, in order to enable feedback loops in response to changes.

## Software Engineering for ML - Overview

- Milestone 1 - Project inception: ML problem spec, model cards, dataset cards, project coordination and communication
- Milestone 2 - Model building: reproducibility: project structure, versioning, experiment tracking
- Milestone 3 - Model building: quality assurance: energy efficiency, static analysis, testing
- Milestone 4 - Model deployment: API development: physical architecture, API development
- Goal: to teach MLOps practices and provide hands-on experience with MLOps tools

## Purpose

- Discussing and complementing the content of the lecture
- Gathering experience in software engineering and software quality for data science and ML projects
- Learning to:
- build ML models components following software engineering practices
- deploy ML models components following software engineering practices

## Organization - Overview

- Students will build teams of approximately 5 members
- During the semester, each team will be responsible for taking part in a data science project, discussing the data analysis, documenting and presenting their results:
- The team will present a report, written in English, summarizing the main aspects of the practice, for example, the process of building an ML component of an ML-based system, and an evaluation of the accuracy of the models and algorithms used.
- The resulting software, duly documented, will be uploaded to a repository.

## Organization - Weekly milestones and feedback

- Teams are expected to:
- Solve the assigned tasks planned for each week.
- Document their results according to a predefined template.
- Present their solutions during the exercise class.
- Students and lecturer will discuss the proposed solutions together in the exercise class.
- Teams are expected to enhance their solutions based on the previous discussions.

## Organization - Project goal

- Each team has to build and deploy an ML component following software engineering best practices.
- Each team is free to choose their own ML component goal (either eliciting their own ML component or defining one from the initial ideas):
- Tip: think in the ML component deployed and used in operation
- In the same group, teams cannot repeat the very same goal (recommended to use a unique dataset)
- First-in, first-out!

## Organization - Goal and datasets suggestions

- Computer vision / image classification. Datasets:
- [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.sciencedirect.com/science/article/abs/pii/S0893608012000457)
- [MNIST](https://ieeexplore.ieee.org/document/6296535)
- [Cifar-10](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf)
- Natural language processing / text classification: Datasets:
- [Sentiment140](https://www-cs-faculty.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Amazon Reviews](https://aclanthology.org/D19-1018/)
- [IMDB Movie Reviews](https://aclanthology.org/P11-1015.pdf)
- Code generation, summarization, etc. Datasets:
- [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)
- [GitHub Code](https://huggingface.co/datasets/codeparrot/github-code)
- [rStar-Coder](https://huggingface.co/datasets/microsoft/rStar-Coder)
- Or suggest to the lecturer:
- Another ML goal for your chosen ML application (examples of ML applications)
- Another dataset, for instance from Kaggle or Hugging Face
- Another dataset or ML model in which you have previously worked

## Organization - Scope

- We focus on quality aspects of building the ML model (e.g., reproducibility, quality assurance) and deploying it as an ML component (e.g., API), rather than a whole ML-based system
- Example architecture (3-layer system, MLOps scope highlighted):
  - Presentation Layer: User Interface (Mobile App Front-end, Unity Bundle) <-> Input/Output Data
  - Logic Layer: Business Logic / Camera Device (Mobile App Back-end, Unity Bundle) <-> Data Preprocessing (Image Processing, Unity & C#) <-> DL Component (Embedded SmallCNN & ResNet34, ONNX Model)
  - Data Layer: Database (collection of new tagged images, local disk)
  - MLOps scope (highlighted box) covers: Data Collection (Dataset, e.g. GTSRB) -> Trained Models (SmallCNN & PyTorch, ResNet34) -> DL Component -> Updated Models (SmallCNN & PyTorch, ResNet34), feeding back into Data Preprocessing and the Database

## Organization - Documentation

- Main report: a single pdf file with all the information
- To be uploaded in Atenea (template in Atenea)
- A cover page clearly stating:
- Name of the team/project
- Which deliverable (initial or final)
- Links to resources: GitHub repository, other resources (e.g., Taiga, Trello)
- Contact mail (or whatever means) for the whole team
- Team members' description (surname, name, UPC e-mail, GitHub account)
- Report structure:
1. Introduction: goal of the project (with success criteria), teammates' evaluation (by default, all happy)
2. Methodology: a subsection with an individual description for each software engineering practice
3. Self-evaluation of the project (retrospective): main challenges/barriers/opportunities, what was learned, what would be done differently, what concepts were used from which courses

Very important:
- Maximum of 15 pages for the initial report, and 30 pages for the final report.
- Update previous sections if necessary (e.g., model card).

## Organization - Software and Replication package

- The resulting software, duly structured, will be uploaded to a repository (preferably a link inside to a public repository inside our GitHub organization)
- [https://github.com/mlops-2526q1-mds-upc](https://github.com/mlops-2526q1-mds-upc)
- Contact: santiago.del.rey@upc.edu and matias.martinez@upc.edu
- Repo name: `MLOps_<name of the team>`
- To foster correctness and reproducibility, it has to follow this project structure (justified deviations allowed!): https://drivendata.github.io/cookiecutter-data-science

## Organization - Communication

- General questions about the documentation or technical issues (git, dvc errors, configuration of the VM, etc.)
- Use the forum 'FAQ' in Atenea.
- Check if your question has already been discussed. If not, create a new discussion for your question.
- Team-specific doubts or any other business
- Send an email to the professor.
- Every email sent about MLOps should have as prefix: `[MLOps] <team name>` (square brackets included)
- Kindly provide some mechanism to your teacher so that s/he can send an email and all the members of the team receive this email
- Every document or repository should have as the prefix `MLOps_<team name>` and then a self-explanatory name (e.g., Initial report, final report, etc.)
- Inform your teachers about any team problem you may experience as soon as possible

## Schedule - Laboratory

| Session | Assignments / Outputs for each laboratory session | Tools                                                     | Date    | Delivery (one day before the session) |
|---------|-----------------------------------------------------|------------------------------------------------------------|---------|-----------------------------------------|
| 1       | Milestone 1 - Project kick-off and inception.        | GitHub repository creation (for model and dataset cards)   | Sep. 9  |                                          |
| 2       | Milestone 2a - Model building: reproducibility       | Git with GitHub Flow, DVC                                  | Sep. 16 |                                          |
| 3       | Milestone 2b - Model building: reproducibility       | MLFlow                                                      | Sep. 23 |                                          |
| 4       | Milestone 3a - Model building: quality assurance     | Pylint, Pytest, Great Expectations                          | Sep. 30 |                                          |
| 5       | Milestone 3b - Model building: quality assurance     |                                                              | Oct. 7  |                                          |
| 6       | Presentation Milestones 1-3                          |                                                              | Oct. 14 | 1st (Milestones 1-3)                    |
| 7       | Milestone 4a - Model deployment: API                 | ML serving (Virtech, AWS,…)                                 | Oct. 21 |                                          |
| 8       | Milestone 4b - Model deployment: API                 | Fast API                                                    | Oct. 28 |                                          |
| -       | There is no class                                    |                                                              | Nov. 4  |                                          |
| 9       | Milestone 5a - Model deployment: Model packaging     | Docker and Docker Compose                                   | Nov. 11 |                                          |
| 10      | Milestone 5b - Model deployment: Model packaging     | GitHub Actions                                               | Nov. 18 |                                          |
| 11      | Milestone 6a - Monitoring                            | Better Uptime, Prometheus, and Grafana, Alibi Detect         | Nov. 25 |                                          |
| 12      | Milestone 6b - Monitoring                            |                                                              | Dec. 2  |                                          |
| 13      | Presentation Milestones 4-6                          |                                                              | Dec. 9  | 2nd (Milestones 4-6)                    |
| 14      | Review Milestones 4-6                                |                                                              | Dec. 16 |                                          |

## Schedule - Team

- The team:
- Identified by a name chosen by students
- Ideally 5 members
- One member of the team sends an e-mail to the lecturers, with cc the whole team
- Subject: `[MLOps] <name of the team>`
- Content: name, surname, GitHub username, and UPC e-mail of each member; proposal of ML component to develop and dataset (it can be refined until the 2nd week)
- Deadline: Team created in the first laboratory session!!

## Milestone 1: Inception

- Each team chooses and downloads a dataset for the ML component
- Each team defines a dataset card and model card about their ML component
- Each team creates its collaborative working space (e.g., GitHub repository, Jupyter Notebook, a Discord server…)

| Practice                                                  | Tool(s)                                    |
|------------------------------------------------------------|----------------------------------------------|
| Selection of problem and requirements engineering for ML   | Model and dataset cards (by Hugging Face)     |
| Project coordination and communication                     | Taiga, Trello, Slack                          |

Evaluation criteria:
- Goal Definition and Requirements Engineering: how clearly does your project definition and requirements show what makes your system unique and complete? How well does your dataset and model documentation communicate the choices you made and their implications for others?
- Project Coordination and Communication: how effectively do your coordination and communication practices support collaboration and leave a useful trace of your progress?

## Milestone 2: Model building - reproducibility

- Each team applies practices for reproducibility of the project: project structure, code versioning, data versioning, experiment tracking

| Practice                  | Tool(s)                             |
|----------------------------|---------------------------------------|
| Project structure          | Cookiecutter data science template    |
| Code and data versioning   | Git with GitHub Flow, DVC             |
| Experiment tracking        | MLflow                                |

Evaluation criteria:
- Project Structure: how well does your project structure support clarity, maintainability, and adaptation to your project's needs?
- Code and Data Versioning: how do your version control practices ensure collaboration is smooth and mistakes are minimized? How effectively does your data versioning strategy guarantee reproducibility and scalability for your project?
- Experiment Tracking: how does your experiment tracking help you and others compare, understand, and extend your results?

## Milestone 3: Model building - quality assurance

- Each team applies practices for quality assurance of the project: reporting CO2 emissions, static analysis (e.g., Pynblint linter), testing model, testing data

| Practice                                                            | Tool(s)                                                                               |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Energy efficiency awareness                                          | CodeCarbon                                                                                |
| Quality assurance for ML (static analysis + testing data and model)  | Pynblint (notebook + repository QA), Pylint or flake8, Pytest and Great Expectations      |

Evaluation criteria:
- Energy Efficiency Awareness: how do your energy tracking practices help you reflect on the efficiency and sustainability of your choices?
- Quality Assurance for ML: to what degree does your use of linting contribute to the overall quality and consistency of your codebase? To what extent do your tests give confidence that your code and data behave as expected under different conditions? How effectively does your data validation strategy anticipate and prevent potential issues in your pipeline?

## Milestone 4: Model deployment - API

- Select the cloud provider where you will deploy your ML component (a virtual machine from FIB is available for each team if necessary)
- Each team designs the deployment of the model as an ML component
- The ML component is available via an API and is tested

| Practice          | Tool(s)                                                                    |
|---------------------|--------------------------------------------------------------------------------|
| ML system design    | Cloud platform selected by the students (VMs, Heroku, DigitalOcean, etc.)       |
| APIs for ML         | FastAPI, Pytest (to test the API endpoints)                                     |

Evaluation criteria:
- ML System Design: in what ways does your system design balance clarity and justified complexity while applying suitable architectural patterns and principles?
- APIs for ML: to what extent does your API design, together with its documentation, make the system robust, intuitive, and easy to adopt by others?

## Milestone 5: Model deployment - component delivery

- Each team creates a component ready to be delivered and deployed
- Each team defines a process for automatizing the ML pipeline

| Practice                    | Tool(s)          |
|-------------------------------|---------------------|
| Container and Orchestration   | Docker              |
| CI/CD for ML                  | GitHub Actions      |

Evaluation criteria:
- Container and Orchestration: in what ways does your containerization approach guarantee portability, modularity, consistency, and efficiency across environments?
- CI/CD for ML: what aspects of your CI/CD pipeline demonstrate that automation and collaboration are fully integrated into your process?

## Milestone 6: Monitoring

- Each team has to design and implement solutions for monitoring: resources, model performance

| Practice             | Tool(s)                |
|------------------------|---------------------------|
| Resource Monitoring    | Grafana, Prometheus       |
| Model Performance       | Alibi Detect               |

Evaluation criteria:
- Resource Monitoring: to what extent does your monitoring anticipate real-world challenges and provide actionable insights into system performance?
- Model Performance: to what extent does your monitoring provide timely, actionable signals about data changes, model performance degradation, and user impact, enabling confident investigation and response?

## Schedule - Feedback and Presentations

- Presentations after every three milestones.
- M1-M3 presentation (15 mins), 6th week - Oct. 14: focus on inception (ML component definition and success criteria), and model building: reproducibility & quality assurance
- M4-M6 presentation (15 mins), 13th week - Dec. 9: focus on model deployment: API & CI/CD and monitoring

## Schedule - Deliveries

- 1st report (Milestones 1-3): Oct. 13, 23:55 - via Atenea
- 2nd report (Milestones 4-6): Dec. 8, 23:55 - via Atenea
- What: current/final version of the team report (max 15 pages for the first report, 30 pages for the second), written in English, summarizing the main aspects of the practice, plus a link to the code repository and the presentation slides.

## Lab Evaluation

- Laboratory grade: sum of SE practices scores * Individual factor
- Assessed based on: the report, the replication package (the software repository), and the discussions in the laboratory (including presentations)

| Practice                                                             | Tool(s)                                                                               | Points (out of 100) |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|------------------------|
| Selection of problem and requirements engineering for ML                | Model and dataset cards (by Hugging Face)                                                 | 6                      |
| Project coordination and communication                                  | Taiga, Trello, Slack                                                                       | 4                      |
| Project structure                                                       | Cookiecutter data science template                                                        | 5                      |
| Code and data versioning                                                | Git with GitHub Flow, DVC                                                                  | 15                     |
| Experiment tracking                                                     | MLflow                                                                                     | 5                      |
| Energy efficiency awareness                                             | CodeCarbon                                                                                 | 5                      |
| Quality assurance for ML (static analysis + testing data and model)     | Pynblint (notebook + repository QA), Pylint or flake8, Pytest and Great Expectations       | 10                     |
| ML system design                                                        | Software architecture, cloud platform selected by the students (VMs, Heroku, DigitalOcean, etc.) | 15                 |
| APIs for ML                                                              | FastAPI, Pytest (to test the API endpoints)                                                | 10                     |
| Container and Orchestration                                             | Docker                                                                                     | 5                      |
| CI/CD for ML                                                             | GitHub Actions                                                                             | 10                     |
| Resource Monitoring                                                      | Grafana, Prometheus                                                                        | 5                      |
| Model Performance                                                        | Alibi Detect                                                                               | 5                      |

Each practice is evaluated according to these criteria:
- Poor: the students were not able to apply the recommended MLOps practice
- Fair: the students implemented the MLOps practice by replicating the provided examples with minor changes
- Good: the students implemented the MLOps practice by replicating the provided examples with major changes
- Excellent: the students implemented the MLOps practice in an extended or innovative way

## Lab Evaluation: Individual factor

- Individual factor between 0 and 1.2 (exceptional cases); normally 1.0 for all team members (this is a team goal). The whole team has the same base grade.
- The individual factor depends on your contribution to your team, quantified by the teacher based on:
- Peer review by the team members following the CATME rubric
- Individual continuous work and contributions seen by the professor based on: the report, the replication package (number of commits in the main branch and total lines of code changed), and the discussions in the laboratory (including presentations)
- Inform both your team members and your teachers about any team problems you may experience as soon as possible and report them in the deliveries.
- A lower individual factor for a team member does not automatically imply an increase for another member, nor the other way around.
