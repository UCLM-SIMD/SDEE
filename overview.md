# Overview

Predict, given the current state of the project at time tpredtpred​, the difference between the actual delivered velocity and the committed (target) velocity, defined as velocity(Difference):

velocity(Difference)=velocity(Delivered)−velocity(Committed)

For example, in the iteration Mesosphere Sprint 35, the difference between the actual delivered velocity and the committed velocity was -12. Specifically:

velocity(Difference)=−12

This occurred because the delivered velocity was 7, while the committed velocity at tpredtpred​ was 19. In this case, the iteration delivered below the target, i.e.,

velocity(Committed)>velocity(Delivered)

It’s important to note that velocity(Difference)=0velocity(Difference)=0 does not necessarily imply that all commitments (in terms of specific issues to be resolved) were fully met. Rather, it assesses the overall quantum of work performed.

### Apache dataset

- 30% of features
- Top features:
  - Velocity ToDo: -SHAP
  - Velocity at Start Time: -SHAP
  - Frequency of improvement tasks: +SHAP
  - Velocity InProgress: -SHAP
  - Frequency of priority major: +SHAP
  - Frequency of gunning fog hard: +SHAP
  - Frequency of type Story: +SHAP

1. Velocity ToDo (Negative SHAP)

   Explanation: This feature likely represents the amount of work in the "ToDo" state at the beginning of the iteration. A negative SHAP value indicates that when the initial Velocity ToDo is higher, the predicted vel_diff tends to be more negative (i.e., the team fails to meet their committed velocity).
   Reasoning: This could be because if a significant portion of work is left in the ToDo state early on, it suggests a lack of readiness or planning, leading to potential delays or inefficiencies. This makes it harder to achieve the committed velocity, so the difference between committed and delivered velocities increases.

2. Velocity at Start Time (Negative SHAP)

   Explanation: This feature measures the team's velocity at the start of the iteration. A negative SHAP value means that when the Velocity at Start Time is higher, the model predicts a lower vel_diff (underperformance).
   Reasoning: A higher initial velocity might imply overconfidence or mismanagement of work allocation. The team may overcommit based on an optimistic view of their initial velocity, leading to a failure in maintaining that pace throughout the iteration, resulting in a larger negative vel_diff.

3. Frequency of Improvement Tasks (Positive SHAP)

   Explanation: This feature captures the number of improvement-related tasks within an iteration. A positive SHAP value suggests that a higher frequency of improvement tasks is associated with a positive vel_diff (i.e., the team delivers more than they committed).
   Reasoning: Improvement tasks often aim to streamline processes, reduce inefficiencies, or improve code quality. A higher number of such tasks may lead to improved team productivity, thus exceeding the committed velocity by improving the workflow during the iteration.

4. Velocity InProgress (Negative SHAP)

   Explanation: This feature reflects the velocity of tasks that are "In Progress" during the iteration. A negative SHAP value indicates that a higher Velocity InProgress correlates with a lower vel_diff.
   Reasoning: High velocity in the "In Progress" state could signal that many tasks remain incomplete or stuck in progress. This indicates inefficiencies or bottlenecks that prevent tasks from being completed within the iteration, causing the team to fall short of their committed velocity.

5. Frequency of Priority Major (Positive SHAP)

   Explanation: This feature counts the number of high-priority tasks in the iteration. A positive SHAP value suggests that a higher frequency of major priority tasks leads to a higher vel_diff (i.e., the team delivers more than expected).
   Reasoning: Major priority tasks may receive more attention and resources, pushing the team to focus and work more effectively. The presence of critical tasks may drive productivity and lead to overperformance relative to the committed velocity.

6. Frequency of Gunning Fog Hard (Positive SHAP)

   Explanation: This feature measures the frequency of tasks considered difficult based on the Gunning Fog index, which evaluates the complexity of tasks. A positive SHAP value means that more complex tasks correlate with a positive vel_diff.
   Reasoning: Teams might overestimate the difficulty of complex tasks and allocate more time than necessary. Consequently, when the tasks are completed faster than expected, the team exceeds the committed velocity. Additionally, tackling difficult tasks may encourage more focus and better planning, leading to overperformance.

7. Frequency of Type Story (Positive SHAP)

   Explanation: This feature tracks the number of user stories in the iteration. A positive SHAP value indicates that a higher frequency of stories correlates with a higher vel_diff.
   Reasoning: User stories typically represent tangible features or functionality that teams deliver to the end user. Focusing on user stories may lead to higher motivation and clearer goals, enabling the team to deliver more than they committed. Additionally, user stories are often well-scoped and achievable, making it easier to exceed the committed velocity.

Summary:

    Negative SHAP Values (e.g., Velocity ToDo, Velocity at Start Time, Velocity InProgress) suggest that certain features, when higher, lead to a larger negative vel_diff. This could indicate that these features represent signs of inefficiency, poor planning, or bottlenecks that prevent the team from achieving their committed velocity.
    Positive SHAP Values (e.g., Frequency of Improvement Tasks, Frequency of Priority Major, Frequency of Gunning Fog Hard, Frequency of Type Story) suggest that features like improvements, focused priority tasks, and complexity are correlated with better-than-expected performance. This could indicate that these features motivate the team to deliver more or overestimate challenges, resulting in a positive vel_diff.

### JBoss dataset

- 30% of features
- Top features:
  - Velocity ToDo: -SHAP
  - Nº Issues at Start Time: -SHAP
  - Velocity InProgress: -SHAP
  - Nº Issues In Progress: -SHAP
  - Velocity at Start Time: +SHAP
  - Frequency of priority optional: -SHAP
  - Nº Issues In ToDo: -SHAP

1. Velocity ToDo (Negative SHAP)

   Explanation: Similar to the Apache dataset, Velocity ToDo refers to the amount of work in the "ToDo" state at the start of the iteration. A negative SHAP value indicates that when this value is higher, the predicted vel_diff is more negative (i.e., the team is underperforming relative to their committed velocity).
   Reasoning: A large Velocity ToDo likely signals a lack of preparedness or insufficient task refinement. The team may struggle to convert planned tasks into completed work within the iteration, leading to an inability to meet their committed velocity, which results in a negative vel_diff.

2. Nº Issues at Start Time (Negative SHAP)

   Explanation: This feature counts the number of issues present at the start of the iteration. A negative SHAP value indicates that a higher number of issues at the start of the iteration is associated with a lower predicted vel_diff.
   Reasoning: A higher number of issues at the beginning of the iteration can be a sign of poor backlog management or overwhelming task load. This could result in the team being overloaded and unable to complete all the committed tasks, leading to a larger negative velocity difference (vel_diff).

3. Velocity InProgress (Negative SHAP)

   Explanation: This feature represents the velocity of tasks that are "In Progress" during the iteration. A negative SHAP value indicates that higher velocity in progress is associated with a more negative vel_diff.
   Reasoning: High velocity in the "In Progress" state might suggest that tasks are not being completed efficiently and are getting stuck in progress. This might indicate bottlenecks or blockers that prevent the team from completing tasks on time, which leads to a negative velocity difference.

4. Nº Issues In Progress (Negative SHAP)

   Explanation: This feature counts the number of issues that are in the "In Progress" state. A negative SHAP value suggests that when the number of in-progress issues is higher, the vel_diff is predicted to be more negative.
   Reasoning: A higher number of in-progress issues may signal task management inefficiencies. If many tasks are in progress simultaneously, it could indicate a lack of focus or resource contention, preventing the team from completing all of them, resulting in a negative velocity difference.

5. Velocity at Start Time (Positive SHAP)

   Explanation: This feature measures the team's velocity at the beginning of the iteration. A positive SHAP value indicates that higher initial velocity correlates with a more positive vel_diff (i.e., better performance than expected).
   Reasoning: A higher initial velocity could indicate strong momentum at the start of the iteration, leading to a situation where the team is able to exceed their committed velocity by the end of the iteration. This could be due to better planning or efficient task execution early on.

6. Frequency of Priority Optional (Negative SHAP)

   Explanation: This feature captures the frequency of tasks labeled as "priority optional" within the iteration. A negative SHAP value suggests that a higher frequency of optional priority tasks is associated with a more negative vel_diff.
   Reasoning: Optional priority tasks might not be critical, and focusing on these tasks could divert attention away from more important work. If the team spends time on non-essential tasks, they may fail to complete their core commitments, leading to a negative difference between committed and delivered velocity.

7. Nº Issues In ToDo (Negative SHAP)

   Explanation: This feature tracks the number of issues in the "ToDo" state. A negative SHAP value indicates that a higher number of issues in the "ToDo" state correlates with a more negative vel_diff.
   Reasoning: A large number of tasks in the "ToDo" state could suggest that the team has not adequately scoped or prepared their tasks for the iteration. This backlog of work may overwhelm the team, making it difficult to finish all the committed tasks, leading to underperformance relative to the committed velocity.

### Jira dataset

- 30% of features
- Top features:
  - Velocity ToDo: -SHAP
  - Frequency of priority major: +SHAP
  - Frequency of type improvement: +SHAP
  - Velocity at Start Time: -SHAP
  - Velocity InProgress: -SHAP
  - Frequency of type bug: +SHAP
  - Frequency of priority critical: +SHAP

1. Velocity ToDo (Negative SHAP)

   Explanation: This feature likely measures the velocity or the amount of work in the "ToDo" state at the beginning of the iteration. A negative SHAP value indicates that when this value is higher, the predicted vel_diff is more negative (i.e., the team underperforms relative to their committed velocity).
   Reasoning: A higher Velocity ToDo might indicate that a significant portion of tasks is still in the planning or scoping phase. This can suggest poor readiness or delays in starting work, making it difficult for the team to meet their committed velocity. Consequently, more tasks are left incomplete, leading to a negative vel_diff.

2. Frequency of Priority Major (Positive SHAP)

   Explanation: This feature captures the frequency of tasks with "major" priority within an iteration. A positive SHAP value suggests that a higher number of major-priority tasks correlates with a more positive vel_diff (i.e., the team performs better than expected).
   Reasoning: Major priority tasks often require focused attention and resources. The presence of more high-priority tasks might lead to better organization and prioritization, pushing the team to complete their core commitments more efficiently. As a result, the team may exceed their committed velocity, leading to a positive vel_diff.

3. Frequency of Type Improvement (Positive SHAP)

   Explanation: This feature measures the frequency of improvement-related tasks within the iteration. A positive SHAP value means that a higher number of improvement tasks is associated with a more positive vel_diff.
   Reasoning: Improvement tasks typically aim to enhance processes, quality, or performance. When teams focus on improvements, they may streamline workflows, eliminate bottlenecks, or make their work more efficient. This often leads to improved productivity, allowing the team to deliver more than they initially committed to, hence a positive vel_diff.

4. Velocity at Start Time (Negative SHAP)

   Explanation: This feature measures the team's velocity at the start of the iteration. A negative SHAP value indicates that when this value is higher, the predicted vel_diff tends to be more negative.
   Reasoning: Higher initial velocity could signal that the team is front-loading work, but this might not be sustainable throughout the iteration. The team may overcommit based on their initial momentum, leading to underperformance later in the iteration. This overestimation can cause them to fall short of their committed velocity, resulting in a negative vel_diff.

5. Velocity InProgress (Negative SHAP)

   Explanation: This feature reflects the velocity of tasks that are currently "In Progress" during the iteration. A negative SHAP value suggests that a higher Velocity InProgress correlates with a more negative vel_diff.
   Reasoning: High Velocity InProgress might imply that many tasks are stuck in progress and not being completed. This indicates inefficiencies or blockers that prevent tasks from moving through the workflow and being finished. When too many tasks are in progress but not completed, the team struggles to achieve their committed velocity, leading to a negative vel_diff.

6. Frequency of Type Bug (Positive SHAP)

   Explanation: This feature measures the frequency of bug-related tasks during the iteration. A positive SHAP value indicates that a higher number of bug-fixing tasks is associated with a more positive vel_diff.
   Reasoning: Focusing on bugs could indicate that the team is dedicating effort to improving the quality of the product. Successfully fixing bugs may prevent future issues, reduce rework, and enhance team productivity. By addressing bugs early in the iteration, the team might free up resources and time to work on additional tasks, leading to overperformance relative to their commitment, hence a positive vel_diff.

7. Frequency of Priority Critical (Positive SHAP)

   Explanation: This feature captures the frequency of tasks with "critical" priority within the iteration. A positive SHAP value suggests that a higher number of critical priority tasks leads to a more positive vel_diff.
   Reasoning: Critical tasks often demand immediate attention and resources, meaning they are likely to be resolved promptly. This focus on critical issues may improve overall team efficiency and prioritization. When critical tasks are handled well, the team may exceed their committed velocity due to better organization and focus on delivering high-value work, resulting in a positive vel_diff.
