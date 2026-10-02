# LearnAI — Personalized AI Tutor: ML/AI Architecture & Integration Specification

---

## TL;DR FOR ML DEVELOPER

> **What LearnAI is:** LearnAI is an adaptive learning platform that abandons the "one-size-fits-all" approach to teaching Artificial Intelligence. Instead of forcing every student through identical static tutorials, it continuously estimates each learner's knowledge state, diagnoses misconceptions, calibrates difficulty in real time, and dynamically restructures their learning trajectory.
>
> **What you are building:** You are building the **AI / ML intelligence layer** consisting of two core subsystems: (1) a **Context-Aware Personalized AI Tutor Chatbot** that dynamically adjusts pedagogical depth, mathematical rigor, and tone based on learner profile; and (2) an **Adaptive Question Generator & Evaluation Engine** that synthesizes targeted questions (MCQs, debugging, scenarios) keyed to detected knowledge gaps and adjusts difficulty using mastery thresholds.
>
> **Current status:** The frontend is a fully functional React + TypeScript + Tailwind prototype running on local state (`LearnerContext.tsx`) and mock data (`mockLearner.ts`, `mockTutorResponses.ts`, `mockQuestions.ts`). It simulates profile switching, diagnostics, roadmaps, and adaptive threshold events.
>
> **Future integration:** You will expose clean HTTP APIs (recommended: FastAPI) that receive the learner's context payload, run LLM orchestration / prompt pipelines, and return structured JSON responses. The frontend will gracefully swap out mock functions (`getTutorReply()`, static question lists) with your live endpoints without altering UI layout or breaking fallback mock mode.

---

## 1. PROJECT OVERVIEW & PROBLEM STATEMENT

### 1.1 The Core Problem
Conventional computer science and AI education treats every student identically. A junior developer with two years of Python and an absolute beginner with zero coding background receive:
- The exact same lecture videos
- The same mathematical explanations (often either overly opaque or overly dumbed-down)
- The same practice problems
- The same fixed linear curriculum sequence
- The same learning pace

When a learner struggles with a fundamental concept (such as the *Bias-Variance Tradeoff*), traditional platforms simply mark the question wrong and proceed forward, compounding gaps until the student becomes overwhelmed and abandons the course.

### 1.2 The LearnAI Solution
LearnAI transforms learning into an interactive closed-loop feedback system. It diagnoses baseline skills, tracks mastery per concept, detects weaknesses immediately after quizzes, adjusts roadmap milestones, and delivers pedagogical explanations calibrated to the learner's exact cognitive level.

```text
                     ┌───────────────────────────┐
                     │          ASSESS           │ (Diagnostic Questionnaire)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │    UNDERSTAND LEARNER     │ (Goal, Experience, Strengths/Gaps)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │        PERSONALIZE        │ (Generate Custom Path & Focus)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │           TEACH           │ (Level-Calibrated Explanations)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │         PRACTICE          │ (Adaptive Question Generation)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │          MEASURE          │ (Analyze Accuracy & Timings)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │      DETECT WEAKNESS      │ (Identify Failing Concepts)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │           ADAPT           │ (Adjust Difficulty & Inject Revision)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │        TEACH AGAIN        │ (Remediate with New Analogies/Hints)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │     SHOW IMPROVEMENT      │ (Skill Map & Progress Velocity)
                     └───────────────────────────┘
```

---

## 2. SYSTEM ARCHITECTURE & EVOLUTION

### 2.1 Current Frontend Prototype Architecture
The current application runs entirely in the browser:
- State is held in React Context (`src/contexts/LearnerContext.tsx`) and backed by `localStorage` (`learnai_state_v2`).
- The Tutor chatbot uses regex/keyword matching against a static array in `src/data/mockTutorResponses.ts`.
- Practice questions are drawn from a static array in `src/data/mockQuestions.ts`.
- Adaptive decisions are calculated via deterministic heuristics in `recordQuizScore()`.

```text
┌────────────────────────────────────────────────────────┐
│                   React 18 Frontend                    │
│                                                        │
│  ┌─────────────────┐       ┌────────────────────────┐  │
│  │  Pages / Views  │ <───> │  LearnerContext.tsx    │  │
│  │  (Tutor, Learn, │       │  - Active Profile      │  │
│  │  Practice, etc) │       │  - Learning Roadmap    │  │
│  └─────────────────┘       │  - Adaptive Events     │  │
│                            │  - localStorage Sync   │  │
│                            └───────────┬────────────┘  │
│                                        │               │
│               ┌────────────────────────┴────────┐      │
│               ▼                                 ▼      │
│  ┌─────────────────────────┐     ┌──────────────────┐  │
│  │  mockTutorResponses.ts  │     │ mockQuestions.ts │  │
│  │  (Keyword-match reply)  │     │ (Static MCQs)    │  │
│  └─────────────────────────┘     └──────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 2.2 Target ML / Backend Architecture
The ML engineer will construct an AI orchestration backend (FastAPI recommended) that exposes RESTful endpoints to the React frontend.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             REACT FRONTEND (Vite)                           │
│  • UI Rendering & Markdown/KaTeX Formatting                                 │
│  • User Event Dispatch (Chat queries, quiz submissions)                     │
│  • Fallback Mode: Seamless fallback to mock data if backend is offline       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP REST / JSON
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI / ML BACKEND (FastAPI)                         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         API Endpoints Router                          │  │
│  │  /api/tutor/chat  |  /api/questions/generate  |  /api/questions/eval  │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                       │
│  ┌───────────────────────────────────┴───────────────────────────────────┐  │
│  │                    Learner State & Context Builder                    │  │
│  │  Extracts user level, strengths, weaknesses, goal, and recent scores   │  │
│  └───────────────────┬───────────────────────────────┬───────────────────┘  │
│                      │                               │                       │
│                      ▼                               ▼                       │
│  ┌─────────────────────────────────────┐  ┌──────────────────────────────┐  │
│  │      Tutor Orchestration Pipeline   │  │  Adaptive Question Engine    │  │
│  │  • Pedagogical Prompt Formulator    │  │  • Difficulty Calibrator     │  │
│  │  • Explanation Depth Controller     │  │  • Non-Repetition Filter     │  │
│  │  • Checkpoint Question Inserter     │  │  • Reason / "Why" Synthesizer│  │
│  └───────────────────┬─────────────────┘  └──────────────┬───────────────┘  │
│                      │                                   │                   │
│                      └─────────────────┬─────────────────┘                   │
│                                        ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    LLM Provider (OpenAI / Gemini)                     │  │
│  │  • Temperature: 0.2 (Evaluation/Code) / 0.7 (Tutoring/Analogies)      │  │
│  │  • Enforced Pydantic / JSON Structured Outputs                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. APPLICATION ROUTING & ML RELEVANCE

The application routes are defined in [`src/App.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/App.tsx). Below is the categorization of all routes and their direct relevance to the ML subsystems:

| Route | Page Component | Directly ML Relevant? | ML Subsystem Connection |
|---|---|:---:|---|
| `/tutor` | [`Tutor.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Tutor.tsx) | **YES (CRITICAL)** | **Personalized AI Tutor Chatbot**: Sends user query + full learner context payload; streams back adaptive markdown explanations, follow-up pills, and checkpoint quizzes. |
| `/practice` | [`Practice.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Practice.tsx) | **YES (CRITICAL)** | **Question Generator & Evaluator**: Generates adaptive questions on demand; evaluates submitted answers; calculates score and triggers adaptive events. |
| `/assessment` | [`Assessment.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Assessment.tsx) | **YES** | **Baseline Profiler**: Collects initial experience, languages, topics, and goals; initializes baseline mastery scores for the ML engine. |
| `/learning-path` | [`LearningPath.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/LearningPath.tsx) | **YES** | **Dynamic Curriculum**: Visualizes milestones; marks nodes as `adapted`, `completed`, or `current` based on ML performance triggers. |
| `/learn/:lessonId`| [`Learn.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Learn.tsx) | **YES** | **Contextual In-Lesson Tutor**: Embedded sidebar chatbot that answers questions strictly anchored to the active lesson and concept. |
| `/dashboard` | [`DashBoard.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/DashBoard.tsx) | **YES** | **Adaptive Decision Feed**: Displays real-time alert banners explaining *why* the AI adapted the user's path (before/after comparison). |
| `/progress` / `/skills`| [`Progress.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Progress.tsx) | **YES** | **Knowledge State Visualizer**: Radar chart of 6 core skills, weekly accuracy trajectories, and detected weakness modules. |
| `/` | `Home.tsx` | No | Marketing landing page explaining LearnAI capabilities. |
| `/courses`, `/about`, `/contact`, `/playground`, `/solve/:slug` | Supporting pages | Minor | Supplementary code execution playground and directory pages. |

---

## 4. LEARNER STATE ARCHITECTURE ([`LearnerContext.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/contexts/LearnerContext.tsx))

The `LearnerContext` is the centralized state store of the frontend. It manages active state and provides methods that manipulate the learner's journey.

### 4.1 State Variables
1. **`profile: LearnerProfile`**: The comprehensive learner record containing demographics, goals, mastery scores, strengths, and weaknesses.
2. **`activeProfileId: string`**: `"beginner" | "intermediate" | "advanced"` — indicates which demo persona is active.
3. **`learningPath: PathNode[]`**: The sequence of curricular milestones, their completion percentages, and adaptation flags.
4. **`adaptiveEvents: AdaptiveEvent[]`**: Chronological log of real-time adaptation events triggered by quiz scores.
5. **`completedLessons: string[]`**: Array of lesson IDs already finished by the learner.

### 4.2 Core Context Functions & Future ML Mapping

| Function | Current Behavior in Prototype | What Needs to be Sent to ML Backend |
|---|---|---|
| `switchDemoProfile(id)` | Loads preset profile (`beginner`, `intermediate`, or `advanced`) from `mockLearner.ts` and resets path. | Send profile ID or load learner session from database. ML tutor/generator immediately calibrates to persona. |
| `recordQuizScore(topic, scorePercent, totalQuestions, category)` | Evaluates score against thresholds (`<60`, `60-84`, `>=85`), updates skill radar percentage, modifies path node status, and creates an `AdaptiveEvent`. | **Critical Performance Event**: Send `{ topic, scorePercent, totalQuestions, category, questionIds, errorBreakdown }`. ML backend recalculates Bayesian knowledge state and returns adaptation decisions. |
| `applyAssessment(data)` | Synthesizes answers from 6-step questionnaire, sets experience level, identifies initial strengths/weaknesses, and formats `whyThisPath` string. | Send complete questionnaire answers. ML backend runs baseline profiling prompt/model, creates user profile, and constructs a customized curriculum path. |
| `markLessonComplete(lessonId)` | Adds `lessonId` to `completedLessons`, sets node status to `completed` (100%), and increments concepts mastered. | Send `{ lessonId, timestamp }`. ML updates prerequisite graph to unlock subsequent topics. |
| `toggleDailyPlanItem(planId)` | Checks/unchecks daily review or practice checklist items. | Send `{ planId, completed }` to update engagement telemetry and streak tracking. |
| `resetToDefault()` | Clears `localStorage` and restores intermediate baseline state. | Flushes testing cache/session in ML backend. |

---

## 5. THE LEARNER PROFILE DATA MODEL ([`mockLearner.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/data/mockLearner.ts))

The `LearnerProfile` interface defines everything known about a user. Every field serves a distinct purpose in the ML systems.

### 5.1 Learner Profile Field Mapping Table

| Field | Type | Meaning & Purpose | Used by Chatbot? | Used by Question Generator? | Used for Adaptation? | Why the ML System Needs It |
|---|---|---|:---:|:---:|:---:|---|
| `id` | `string` | Unique learner identifier | Yes | Yes | Yes | Identifies session, indexes memory, and tracks history. |
| `name` | `string` | Learner's display name | Yes | No | No | Enables warm, personalized conversational greetings. |
| `level` | `"Beginner" \| "Intermediate" \| "Advanced"` | High-level cognitive & technical bracket | **Yes** | **Yes** | **Yes** | **Primary anchor for tone, mathematical depth, code syntax, and initial question difficulty.** |
| `role` | `string` | Current professional role | Yes | No | No | Allows contextual analogies (e.g. comparing models to backend APIs for a developer). |
| `goal` | `string` | Stated learning objective (e.g., "Become an ML Engineer") | **Yes** | **Yes** | **Yes** | Prioritizes target domains; filters out unneeded theoretical tangents. |
| `targetRole` | `string` | Desired future career title | Yes | Yes | Yes | Focuses questions on interview-style or production-grade challenges. |
| `experience` | `string` | Baseline technical background summary | Yes | Yes | Yes | Informs the tutor what foundational programming concepts can be assumed. |
| `knownLanguages` | `string[]` | Languages known (e.g. `["Python", "SQL"]`) | Yes | Yes | No | Directs code examples to known languages; prevents unfamiliar syntax confusion. |
| `knownTopics` | `string[]` | Topics learner has already studied | **Yes** | **Yes** | **Yes** | **Prevents asking trivial questions on mastered topics; allows building on prior concepts.** |
| `targetTopics` | `string[]` | Concepts user must master to achieve goal | Yes | **Yes** | **Yes** | Defines the target pool for dynamic question generation. |
| `learningPace` | `"Relaxed" \| "Balanced" \| "Intensive"` | Preferred velocity | Yes | Yes | Yes | Determines question frequency and length of explanations. |
| `dailyCommitmentMinutes` | `number` | Daily study time budget | No | Yes | Yes | Sizes daily problem batches (e.g. 3 questions vs 10 questions). |
| `streakDays` | `number` | Daily login streak count | Yes | No | No | Contextual motivation / positive reinforcement in chat. |
| `overallProgress` | `number` (0–100) | Overall curriculum completion percentage | Yes | No | Yes | High-level measure of trajectory progression. |
| `accuracyRate` | `number` (0–100) | Historical accuracy on all questions solved | **Yes** | **Yes** | **Yes** | Macro-indicator of learner reliability and confidence. |
| `questionsSolved` | `number` | Total number of practice questions answered | No | Yes | Yes | Sample size metric: low count means uncertainty is high; high count means stable mastery. |
| `conceptsMastered` | `number` | Total milestone topics certified | Yes | Yes | Yes | Curriculum velocity metric. |
| `weeklyHoursSpent` | `number` | Hours invested this week | No | No | Yes | Engagement indicator for pace recommendations. |
| `estimatedWeeksRemaining`| `number` | Projected time to goal completion | Yes | No | Yes | Dynamic scheduling adjustment. |
| `whyThisPath` | `string` | Human-readable explanation of why roadmap was tailored | Yes | No | Yes | Transparent explanation displayed on UI; can be quoted by tutor. |
| `strengths` | `Array<{ name, score, description }>` | Concepts where learner scores $\ge 75\%$ | **Yes** | **Yes** | **Yes** | Informs tutor to bridge new topics by connecting them to existing strengths. |
| `weaknesses` | `Array<{ name, score, reason, recommendedLessonId }>` | Concepts where learner scores $< 65\%$ | **Yes (CRITICAL)** | **Yes (CRITICAL)** | **Yes (CRITICAL)** | **Primary driver of remediation: tutor emphasizes these; generator targets them with high priority.** |
| `skills` | `{ [category: string]: number }` | Skill domain scores (Python, Statistics, ML, DL, NLP, GenAI) | **Yes** | **Yes** | **Yes** | Granular skill vector (0–100) used for radar chart and sub-domain difficulty calibration. |
| `weeklyAccuracyTrends` | `Array<{ week, accuracy, studyHours }>` | Historical weekly performance | No | No | Yes | Trend analysis: detects if learner is accelerating, plateauing, or regressing. |
| `dailyPlan` | `Array<{ id, title, type, durationMinutes, completed }>` | Micro-tasks for the current day | Yes | Yes | Yes | Chatbot recommends immediate next tasks from this list. |
| `recentAdaptiveDecision`| `AdaptiveDecision` object | Most recent automated adjustment triggered | **Yes** | **Yes** | **Yes** | Explains recent roadmap mutations so tutor can contextualize ongoing drills. |

---

## 6. DEMO PROFILES DEEP-DIVE

The frontend contains three distinct predefined personas in `DEMO_PROFILES`. The ML system must treat these personas with completely different pedagogical and evaluative strategies:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DEMO PERSONA MATRIX                                    │
├──────────────────┬──────────────────────┬───────────────────────────────────────┤
│ Persona / Level  │ Primary Goal         │ Pedagogical Calibration               │
├──────────────────┼──────────────────────┼───────────────────────────────────────┤
│ ALEX             │ Learn ML from        │ • Analogies & zero-jargon explanations│
│ Beginner         │ Scratch              │ • Low mathematical depth              │
│ (28% Progress)   │                      │ • Code syntax breakdown               │
│                  │                      │ • Foundational MCQs with rich hints   │
├──────────────────┼──────────────────────┼───────────────────────────────────────┤
│ AKSHAT           │ Become an ML         │ • Balanced engineering & math balance │
│ Intermediate     │ Engineer             │ • Overfitting/Regularization focus    │
│ (64% Progress)   │                      │ • Conceptual + code debugging drills  │
│                  │                      │ • Targeted remediation on weak spots  │
├──────────────────┼──────────────────────┼───────────────────────────────────────┤
│ ELENA            │ Build GenAI Apps     │ • Mathematical rigor & system design  │
│ Advanced         │ & Fine-tune LLMs     │ • Transformer mechanics & RAG chunking│
│ (88% Progress)   │                      │ • Complex architectural tradeoffs     │
│                  │                      │ • Hard scenario & edge-case questions │
└──────────────────┴──────────────────────┴───────────────────────────────────────┘
```

### 6.1 Profile 1: Alex (Beginner)
- **Background**: Aspiring tech learner with basic Python syntax (variables, loops). Zero formal statistics or matrix algebra.
- **Skills**: Python: 65%, Statistics: 38%, ML: 25%, DL: 10%, NLP: 5%, GenAI: 10%.
- **Weaknesses**: Matrix Multiplication & Shapes (42%), Standard Deviation & Variance (50%).
- **Tutor Chatbot Behavior**:
  - Use everyday analogies (e.g. cooking, sports, navigation).
  - Do NOT write raw LaTeX equations unless immediately translated into plain English.
  - Explain errors gently with step-by-step reasoning.
- **Question Generator Behavior**:
  - Focus on definitions, core intuition, and straightforward syntax.
  - Always include supportive hints.
  - Options should have clearly distinct conceptual distractors.

### 6.2 Profile 2: Akshat (Intermediate)
- **Background**: Software developer with 2+ years of programming in Python, JS, and SQL. Knows NumPy, Pandas, and basic linear regression.
- **Skills**: Python: 88%, Statistics: 62%, ML: 71%, DL: 35%, NLP: 20%, GenAI: 15%.
- **Weaknesses**: Bias vs Variance Tradeoff (54%), Gradient Descent Optimization (61%), Confusion Matrix & ROC-AUC (63%).
- **Tutor Chatbot Behavior**:
  - Focus on practical implementations (`scikit-learn`, `numpy`).
  - Discuss algorithmic tradeoffs (computational cost vs variance).
  - Use medium mathematical notation (loss functions, partial derivatives).
- **Question Generator Behavior**:
  - Focus on model diagnostics (interpreting loss curves, diagnosing under/overfitting).
  - Include code snippet analysis and hyperparameter tuning scenarios.

### 6.3 Profile 3: Elena (Advanced)
- **Background**: Senior Data Scientist with 4+ years experience. Deep knowledge of PyTorch, CNNs, backprop, and classical ML.
- **Skills**: Python: 98%, Statistics: 94%, ML: 92%, DL: 88%, NLP: 85%, GenAI: 74%.
- **Weaknesses**: Vector DB Embedding Chunking & Semantic Drift (68%).
- **Tutor Chatbot Behavior**:
  - Highly technical, peer-to-peer engineering tone.
  - Deep-dive into paper-level mechanics (e.g., FlashAttention, LoRA rank decomposition, cross-encoders).
  - Skip introductory disclaimers and basic syntax explanations.
- **Question Generator Behavior**:
  - Hard architectural design questions, mathematical derivations, edge cases.
  - Multi-step reasoning questions with subtle distractor options.

---

## 7. ASSESSMENT & DIAGNOSTIC SYSTEM ([`Assessment.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Assessment.tsx))

The diagnostic questionnaire is a 6-step wizard that establishes the learner's initial knowledge state:

```text
Step 1: Experience Level ────► Beginner | Some Programming | Intermediate | Advanced
Step 2: Programming Langs ───► Python | JavaScript | C++ | Java | None
Step 3: AI Topics Known ─────► NumPy, Pandas, Stats, ML, DL, Transformers, RAG, etc.
Step 4: Target Goal ─────────► ML Engineer | Interviews | GenAI Apps | ML from Scratch
Step 5: Learning Pace ───────► Relaxed | Balanced | Intensive
Step 6: Daily Commitment ────► 15 min | 30 min | 45 min | 90+ min
```

### 7.1 How Assessment Influences the Learner Profile
In `applyAssessment(data: AssessmentData)`:
1. **Level Deduction**: Beginner if experience is "Beginner" or topics $\le 1$; Advanced if experience is "Advanced" or topics $\ge 6$; otherwise Intermediate.
2. **Strength Identification**: If "Python" is selected $\rightarrow$ Python baseline set to 85%; if "NumPy/Pandas" selected $\rightarrow$ Data Manipulation set to 80%.
3. **Weakness Deduction**: If "Statistics" is not checked $\rightarrow$ Injects "Applied Statistics" gap (Score: 50%); if "Machine Learning" is not checked $\rightarrow$ Injects "Bias vs Variance" gap (Score: 55%).
4. **Curriculum Pruning (`whyThisPath`)**: Automatically generates an explanation explaining which foundational modules were bypassed because of prior knowledge and why specific modules were scheduled first.

### 7.2 Future ML Baseline Integration
Instead of hardcoded `if/else` rules, the ML system will receive the raw assessment JSON and execute a cold-start profile estimation:
- Synthesize an initial **Knowledge Vector** (probabilities of mastery across 20+ fine-grained AI concepts).
- Generate a customized sequential roadmap DAG (Directed Acyclic Graph) connecting current knowledge to the target goal.
- Generate the personalized rationale text for `whyThisPath`.

---

## 8. ADAPTIVE ENGINE & PERFORMANCE THRESHOLDS

The core adaptation rules are implemented in [`LearnerContext.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/contexts/LearnerContext.tsx#L153-L246) inside `recordQuizScore()`.

### 8.1 Performance Thresholds & Actions

```text
         SCORE < 60%                     60% ≤ SCORE < 85%                    SCORE ≥ 85%
       [STRUGGLING]                         [ON TRACK]                      [ACCELERATING]
             │                                   │                                 │
             ▼                                   ▼                                 ▼
   • Action: "reduced"                 • Action: "maintained"            • Action: "increased"
   • Reduce difficulty of              • Maintain current                • Fast-track roadmap:
     next questions                      difficulty pace                   skip redundant drills
   • Inject prerequisite revision      • Proceed with standard           • Unlock advanced nodes
     module into roadmap                 targeted practice                 early
   • Flag path node as 'adapted'       • Maintain curriculum node        • Advance next node to
   • Set AI Tutor remediation alert      sequence                          'current'
```

### 8.2 The `AdaptiveEvent` Schema
Every quiz submission produces an `AdaptiveEvent` that is displayed in the UI notification feed and the Dashboard alert banner:

```typescript
export interface AdaptiveEvent {
  id: string;              // Unique event ID, e.g. "adapt-1727800000"
  timestamp: string;       // Human-readable time, e.g. "Today at 2:15 PM"
  topic: string;           // Concept tested, e.g. "Bias vs Variance"
  score: number;           // Percentage achieved, e.g. 54
  action: "reduced" | "maintained" | "increased";
  reason: string;          // Concise explanation of why the decision triggered
  recommendation: string;  // Concrete recommendation for the learner
  pathAdjustment: string;  // Description of how the roadmap was altered
}
```

### 8.3 Skill Score Recalculation Formula
Currently, the category skill score is updated using an Exponential Moving Average (EMA):
$$\text{Skill}_{\text{new}} = \text{round}\left(\text{Skill}_{\text{current}} \times 0.70 + \text{Score}_{\text{quiz}} \times 0.30\right)$$
Overall accuracy is updated across total cumulative questions:
$$\text{Accuracy}_{\text{new}} = \text{round}\left(\frac{\text{Accuracy} \times Q_{\text{solved}} + \text{Score}_{\text{quiz}} \times Q_{\text{test}}}{Q_{\text{solved}} + Q_{\text{test}}}\right)$$

---

## 9. CHATBOT SPECIFICATION: PERSONALIZED AI TUTOR

### 9.1 Existing Mock Chatbot Analysis
In [`src/pages/Tutor.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Tutor.tsx) and [`src/data/mockTutorResponses.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/data/mockTutorResponses.ts):
- Receives user string `userMessage`.
- Converts to lowercase and checks if any item in `TUTOR_KNOWLEDGE_BASE` has matching `keywords`.
- Returns hardcoded markdown string, static `suggestedFollowUp` array, and an optional static `practiceQuestionPrompt`.
- Fallback returns a generic placeholder template.

**Why this must be replaced:** The mock response has zero awareness of whether the user is Alex (Beginner) or Elena (Advanced). Both receive the exact same explanation of overfitting or gradient descent!

### 9.2 The Future Context-Aware AI Tutor
The ML-powered chatbot must ingest the user's complete profile and active learning state before formulating a response.

#### Adaptive Explanation Depth: The Contrast Example
When a user asks: *"What is gradient descent?"*

| Dimension | Beginner (e.g. Alex) | Advanced (e.g. Elena) |
|---|---|---|
| **Analogy** | Hiking down a foggy mountain blindfolded, tapping the terrain with a walking stick to find the downhill slope. | Navigating non-convex loss manifolds in $D$-dimensional parameter space. |
| **Math** | Zero raw calculus. Plain English: $\text{New Guess} = \text{Old Guess} - \text{Step Size} \times \text{Slope}$. | Full notation: $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$, discussing second-order Hessian approximations. |
| **Focus** | Why learning rate matters (too big = overshoots; too small = never finishes). | Saddle point traversal, stochastic gradient variance, momentum decay, and AdamW weight decay decoupling. |
| **Code** | 5 lines of Python with print statements for $f(w) = w^2$. | PyTorch custom `torch.optim.Optimizer` step implementation or autograd hook. |
| **Checkpoint** | "If the slope is positive, which way do we step?" | "Why does pure SGD escape saddle points more reliably than full-batch GD?" |

### 9.3 Tutor Learning Modes
The chatbot must flexibly handle different learner prompts while maintaining profile calibration:
1. **Explain (`mode: "explanation"`)**: Core pedagogical concept breakdown.
2. **Simplify (`mode: "simplify"`)**: Re-explains the preceding concept using simpler metaphors, avoiding jargon.
3. **Example (`mode: "example"`)**: Grounds the concept in a practical production use case (e.g., fraud detection, recommendation systems).
4. **Code (`mode: "code"`)**: Delivers executable Python code (NumPy, Scikit-Learn, PyTorch) with line-by-line comments.
5. **Practice (`mode: "practice"`)**: Formulates an immediate checkpoint question to test retention.
6. **Hint (`mode: "hint"`)**: Provides an incremental conceptual nudge on a question without disclosing the answer.
7. **Evaluate (`mode: "evaluate"`)**: Grades a student's open-ended reasoning and explains errors constructively.
8. **Revision (`mode: "revision"`)**: Summarizes the learner's logged weak concepts before an upcoming evaluation.
9. **Learning Path (`mode: "path"`)**: Answers *"Why am I learning this topic next?"* citing roadmap rationale.

---

## 10. QUESTION GENERATOR SPECIFICATION

### 10.1 Existing Mock Question Analysis
In [`src/data/mockQuestions.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/data/mockQuestions.ts), questions follow the `PracticeQuestion` interface:

```typescript
export interface PracticeQuestion {
  id: string;               // e.g. "ml-q1"
  category: "Python" | "Statistics" | "Machine Learning" | "Deep Learning" | "NLP" | "Computer Vision" | "Generative AI" | "LLMs";
  difficulty: "Easy" | "Medium" | "Hard";
  conceptTested: string;    // e.g. "Bias vs Variance"
  title: string;            // e.g. "Identifying Model Behavior from Learning Curves"
  question: string;         // Detailed problem statement
  codeSnippet?: string;     // Optional Python snippet
  options: string[];        // Array of 4 answer options
  correctIndex: number;     // 0, 1, 2, or 3
  explanation: string;      // Comprehensive pedagogical justification
  hint: string;             // Guiding hint without spoiling answer
}
```

### 10.2 Dynamic Question Generator Requirements
The ML Question Generator must NOT sample static files. It must synthesize questions dynamically based on:
1. **Target Topic & Concept**: Anchored to current lesson or detected weakness.
2. **Current Mastery State**: Calibrated to learner's estimated skill level.
3. **Anti-Repetition Mechanism**: Ingests `recent_question_ids` / `recent_concepts` to ensure variety across:
   - *Scenario Variation*: e.g., medical diagnostics $\rightarrow$ financial credit scoring $\rightarrow$ self-driving vehicles.
   - *Question Angle*: e.g., formula interpretation $\rightarrow$ curve diagnosis $\rightarrow$ code debugging.
4. **Why-This-Question Rationale**: Every question must return a clear, user-facing sentence explaining why it was selected. This is vital for judge presentation during hackathons!
   - *Example 1*: *"Generated because you scored 54% on Bias-Variance diagnostics earlier."*
   - *Example 2*: *"Generated as an advanced challenge because you solved 3 consecutive Medium questions correctly."*

### 10.3 Supported Question Archetypes
While the frontend UI currently renders 4-option MCQs cleanly, the schema supports future question formats:
- **Multiple Choice Questions (MCQ)**: Single correct choice out of 4 options (Current primary UI target).
- **Code Debugging**: Learner identifies a bug in a 5–10 line Python snippet.
- **Scenario Diagnostics**: Real-world performance logs / metrics given; learner identifies root failure.
- **Conceptual True/False**: Quick binary checkpoint drills.

---

## 11. END-TO-END DATA FLOW: THE COMPLETE PERSONALIZATION LOOP

Here is how all components interact when a learner uses the platform:

```text
      ┌────────────────────────────────────────────────────────┐
      │  Learner Profile: Intermediate | Goal: ML Engineer     │
      │  Machine Learning: 71% | Bias vs Variance: 54% (Weak)  │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
[1] CHATBOT QUERY ─────────────────────────────────────────────┐
    User: "Why does my model overfit?"                         │
    Tutor receives: Profile + Bias-Variance weakness flag      │
    Tutor returns:                                             │
    • Intermediate explanation of parameter flexibility       │
    • Intuitive polynomial curve analogy                       │
    • Checkpoint Question on Train vs Val loss divergence      │
└─────────────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
[2] PRACTICE QUESTION GENERATION ──────────────────────────────┐
    Generator receives: Topic="Bias vs Variance", Diff="Medium"│
    Generator returns: PracticeQuestion + "why_this_question"  │
    Question: "High train accuracy (99%), low val accuracy (63%)"
└─────────────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
[3] LEARNER SUBMITS WRONG ANSWER ──────────────────────────────┐
    Selected: "High Bias (Underfitting)" (Incorrect)           │
    Correct: "High Variance (Overfitting)"                     │
└─────────────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
[4] PERFORMANCE EVALUATION EVENT ──────────────────────────────┐
    Score: 0% | Concept: "Bias vs Variance"                    │
    Adaptive Engine detects: Score < 60% threshold             │
    Decisions:                                                 │
    1. Action = "reduced"                                      │
    2. Next question difficulty $\rightarrow$ "Easy"          │
    3. Roadmap node 'bias-variance' flagged as 'adapted'       │
    4. AdaptiveEvent created for Dashboard Alert banner        │
└─────────────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
[5] REMEDIATION & RE-ASSESSMENT ───────────────────────────────┐
    Tutor supplies simpler analogy on memorization vs general- │
    ization. Next practice question tests core concept at      │
    "Easy" difficulty. Learner answers correctly.              │
    Mastery updates: 54% ────► 68% ────► 86%                   │
    Progress page displays upward trend in radar chart!        │
└──────────────────────────────────────────────────────────────┘
```

---

## 12. PROPOSED FUTURE API CONTRACTS (FRONTEND ↔ ML BACKEND)

> **Important**: These endpoints represent the target API specifications to be implemented by the ML engineer (e.g. in FastAPI). They will replace the temporary mock functions in the frontend.

### 12.1 Endpoint 1: Tutor Chat

```http
POST /api/v1/tutor/chat
Content-Type: application/json
```

#### Request Payload
```json
{
  "message": "Why does my decision tree overfit?",
  "conversation_id": "conv-101",
  "active_lesson_id": "bias-variance",
  "current_topic": "Decision Trees & Overfitting",
  "learning_mode": "explanation",
  "learner_context": {
    "learner_id": "akshat-intermediate",
    "name": "Akshat",
    "level": "Intermediate",
    "goal": "Become an ML Engineer",
    "learning_pace": "Balanced",
    "known_topics": ["Python", "NumPy", "Pandas", "Linear Regression"],
    "weaknesses": [
      {
        "concept": "Bias vs Variance",
        "current_score": 54
      }
    ],
    "skills": {
      "Python": 88,
      "Statistics": 62,
      "Machine Learning": 71,
      "Deep Learning": 35
    },
    "recent_quiz_score": 54
  }
}
```

#### Response Payload
```json
{
  "success": true,
  "conversation_id": "conv-101",
  "message": "A decision tree overfits when it is allowed to grow without constraints until every single leaf node contains only one sample.\n\n### Why it happens:\nWithout limits like `max_depth` or `min_samples_split`, the tree splits repeatedly on noisy data points, creating hyper-specific boundaries that fail to generalize to unseen test records.",
  "difficulty_level": "Intermediate",
  "concept": "Decision Trees Overfitting",
  "follow_up_suggestions": [
    "How does max_depth prevent overfitting?",
    "Explain Pruning vs Random Forests",
    "Give me an example in Python"
  ],
  "checkpoint_question": {
    "question": "Which parameter in Scikit-Learn's DecisionTreeClassifier is most effective at preventing tree overfitting?",
    "options": [
      "criterion='gini'",
      "max_depth=4",
      "splitter='best'",
      "random_state=42"
    ],
    "correct_index": 1,
    "explanation": "Restricting `max_depth` caps tree height, preventing the model from fitting high-variance leaf artifacts."
  },
  "recommended_next_action": {
    "type": "practice",
    "target_topic": "Tree Regularization",
    "reason": "Reinforcing hyperparameter tuning will remediate your 54% score on variance control."
  }
}
```

---

### 12.2 Endpoint 2: Dynamic Question Generation

```http
POST /api/v1/questions/generate
Content-Type: application/json
```

#### Request Payload
```json
{
  "learner_id": "akshat-intermediate",
  "category": "Machine Learning",
  "target_concept": "Bias vs Variance",
  "difficulty": "Adaptive",
  "count": 3,
  "learner_level": "Intermediate",
  "learner_weaknesses": ["Bias vs Variance", "Gradient Descent"],
  "recent_question_ids": ["ml-q1", "ml-q2"]
}
```

#### Response Payload
```json
{
  "success": true,
  "count": 3,
  "questions": [
    {
      "id": "gen-ml-801",
      "category": "Machine Learning",
      "difficulty": "Medium",
      "concept_tested": "Regularization Hyperparameters",
      "title": "Diagnosing Ridge Regularization Penalty",
      "question": "You train a Ridge regression model and notice that both training loss and validation loss increase when you increase the regularization parameter alpha from 0.1 to 1000. What has occurred?",
      "code_snippet": "from sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1000.0).fit(X_train, y_train)",
      "options": [
        "The model developed extreme High Variance",
        "The model was pushed into severe High Bias (Underfitting)",
        "The learning rate diverged to infinity",
        "The data suffered catastrophic data leakage"
      ],
      "correct_index": 1,
      "explanation": "An excessively large regularization penalty (alpha=1000) penalizes parameter weights too harshly, flattening coefficients toward zero and forcing the model to underfit (High Bias).",
      "hint": "Think about what happens to model flexibility when weights are forced close to zero.",
      "why_this_question": "Generated because your diagnostic test indicated a 54% mastery gap on regularization penalties.",
      "recommended_next_difficulty": "Medium"
    }
  ]
}
```

---

### 12.3 Endpoint 3: Question Evaluation & Adaptation

```http
POST /api/v1/questions/evaluate
Content-Type: application/json
```

#### Request Payload
```json
{
  "learner_id": "akshat-intermediate",
  "question_id": "gen-ml-801",
  "concept_tested": "Regularization Hyperparameters",
  "category": "Machine Learning",
  "selected_option_index": 0,
  "time_taken_seconds": 38,
  "current_category_score": 71
}
```

#### Response Payload
```json
{
  "success": true,
  "is_correct": false,
  "correct_index": 1,
  "score": 0,
  "pedagogical_explanation": "You selected High Variance. However, increasing alpha penalizes model complexity rather than increasing it. Excessively large alpha suppresses coefficients and induces High Bias (Underfitting).",
  "mastery_update": {
    "concept": "Regularization Hyperparameters",
    "previous_score": 54,
    "new_score": 50,
    "category": "Machine Learning",
    "new_category_score": 68
  },
  "adaptive_decision": {
    "action": "reduced",
    "trigger": "Missed question on Regularization Hyperparameters.",
    "reason": "Your score dropped below the 60% threshold on regularization intuition.",
    "recommendation": "We reduced next question complexity to foundational intuition and queued an interactive sandbox.",
    "path_adjustment": "Roadmap Step 5 marked as adapted: Prerequisite intuition module injected."
  }
}
```

---

### 12.4 Endpoint 4: Get/Update Learner Profile

```http
GET /api/v1/learner/profile/{learner_id}
POST /api/v1/learner/assessment
```

Allows synchronization of profile state between React client and ML database.

---

## 13. ERROR HANDLING & RESILIENCE SPECIFICATION

The frontend must **never crash or display a blank screen** if the ML backend experiences timeouts, rate limits, or malformed responses.

### 13.1 Standard Error Contract
If an error occurs on the backend, it must return a standardized JSON error response:

```json
{
  "success": false,
  "error": {
    "code": "MODEL_RATE_LIMIT",
    "message": "The AI service is experiencing high load. Falling back to cached pedagogical response.",
    "details": "Upstream LLM provider 429 quota reached."
  },
  "fallback_data": {
    "reply": "I'm having brief trouble reaching my full neural engine, but remember: Overfitting occurs when your model memorizes sample noise instead of general patterns. Try adding L2 regularization or collecting more training data!",
    "follow_ups": ["Explain L1 vs L2 regularization", "What is cross-validation?"]
  }
}
```

### 13.2 Frontend Fallback Strategy
In the event of network failure or HTTP $5xx$:
1. The frontend catches the exception using `try/catch`.
2. A toast notification informs the user: *"AI service momentarily unreachable. Running in local tutor mode."*
3. The UI smoothly falls back to the deterministic functions in `mockTutorResponses.ts` and `mockQuestions.ts`.
4. User experience remains uninterrupted.

---

## 14. ML TEAM IMPLEMENTATION & LLM ORCHESTRATION GUIDANCE

The ML engineer does **not** need to train a foundation LLM from scratch. The recommended approach is an intelligent orchestration layer built with Python:

### 14.1 Recommended Tech Stack
- **Framework**: FastAPI (asynchronous, high performance, automatic Swagger documentation).
- **LLM Providers**: Anthropic Claude (Claude 3.5 Sonnet) or OpenAI (GPT-4o / GPT-4o-mini) or Google Gemini (Gemini 1.5 Flash).
- **Structured Outputs**: Pydantic models with function calling or `response_format={"type": "json_object"}` to guarantee exact schema conformance.
- **Prompt Engineering**:
  - *System Prompt*: Defines the pedagogical persona, strictly forbidding generic non-personalized answers.
  - *Context Injection*: Dynamic interpolation of `{learner_level}`, `{learner_goal}`, `{learner_weaknesses}`, `{concept_tested}`.
  - *Tone Calibration*: Warm, encouraging, concise, markdown-rich with mathematical precision.

### 14.2 Prompt Engineering Blueprint (Tutor Chatbot)

```text
[SYSTEM PROMPT]
You are LearnAI, an elite, context-aware AI tutor specializing in Artificial Intelligence, Machine Learning, and Data Science.
You are tutoring {learner_name}, who is assessed at the {learner_level} level.
Their primary goal is: "{learner_goal}".
Their known concepts: {known_topics}.
Their detected weaknesses: {weaknesses}.

PEDAGOGICAL RULES:
1. Calibrate your explanation depth strictly to {learner_level}:
   - If Beginner: Use real-world analogies, zero obscure math jargon, step-by-step intuition, friendly encouragement.
   - If Intermediate: Balance practical Python code (scikit-learn/numpy) with mathematical loss functions and tradeoffs.
   - If Advanced: Treat as a peer engineer. Focus on paper-level mechanics, vector complexity, edge cases, and optimization dynamics.
2. Directly address their diagnosed weaknesses ({weaknesses}) when relevant.
3. Always format with clear Markdown headers, bold highlights, and KaTeX math ($...$ or $$...$$).
4. Return a structured JSON response matching the TutorResponseSchema.
```

### 14.3 Retrieval-Augmented Generation (RAG) Consideration
If the team chooses to ground the tutor in specific course textbooks or documentation:
- Store curated AI course material in a lightweight vector database (e.g. ChromaDB, FAISS, or Qdrant).
- Use `text-embedding-3-small` or `bge-small-en-v1.5` to retrieve top-3 relevant textbook chunks.
- Inject retrieved chunks into the prompt as ground truth to prevent hallucination.
- *Note*: RAG is an optional enhancement; strong prompt engineering on frontier models already yields exceptional results for core AI concepts.

---

## 15. SECURITY & ENVIRONMENT CONFIGURATION

1. **API Keys Must NEVER Reside in the Frontend**:
   - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` must **only** be stored in the backend `.env` file.
   - Never prefix LLM keys with `VITE_` in `client/.env`.
2. **CORS Configuration**:
   - Ensure the FastAPI server has `CORSMiddleware` configured to accept requests from the frontend development server (`http://localhost:5173` or `http://localhost:8080`).

---

## 16. CURRENT MOCK VS FUTURE ML COMPARISON

| Feature | Current Prototype (`client`) | Target ML / Backend Implementation |
|---|---|---|
| **Tutor Brain** | Keyword regex search (`mockTutorResponses.ts`) | LLM with dynamic context injection & structured output |
| **Explanation Depth** | Static, identical for all learners | Dynamically calibrated (Beginner analogy $\leftrightarrow$ Advanced math) |
| **Question Source** | 10 static hardcoded MCQs (`mockQuestions.ts`) | On-demand dynamic generation targeted at detected gaps |
| **Question Variety** | Fixed MCQs | Infinite variations across scenarios, code snippets, and concepts |
| **Personalization Rationale** | Hardcoded demo string | LLM synthesizes exact "Why this question was generated" |
| **Adaptation Rules** | Deterministic thresholds in React Context | Bayesian Knowledge Tracing (BKT) / ML scoring engine |
| **Learner Profile Storage** | Browser `localStorage` | PostgreSQL / SQLite database with session persistence |
| **Chat Transcript** | In-memory React state | Backend database logging conversation turns & sentiment |
| **Roadmap Generation** | Static arrays with conditional filters | Dynamically computed Directed Acyclic Graph (DAG) |

---

## 17. PRESERVING FRONTEND STABILITY

> **CRITICAL DIRECTIVE FOR ML INTEGRATION:**
> The current React application is completely functional and demonstrates all views seamlessly. Future integration should be performed by updating API client services (such as [`src/services/api.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/services/api.ts)) to point to backend endpoints, **with automatic fallback to mock data if the backend is down**.
>
> Do not refactor UI components or delete `mockLearner.ts`, `mockQuestions.ts`, or `mockTutorResponses.ts`. They serve as the reliable demo fallback!

---

## 18. FILES THE ML DEVELOPER SHOULD READ FIRST

The files below are ranked in exact order of reading importance:

1. [`src/contexts/LearnerContext.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/contexts/LearnerContext.tsx): **Must Read First.** Contains the state engine, adaptive threshold triggers, quiz recording logic, and profile models.
2. [`src/data/mockLearner.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/data/mockLearner.ts): Defines the `LearnerProfile` interface and the three demo learner personas (`beginner`, `intermediate`, `advanced`).
3. [`src/pages/Tutor.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Tutor.tsx): The chat interface component. Review how messages, follow-up pills, and comprehension check quizzes are rendered.
4. [`src/data/mockTutorResponses.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/data/mockTutorResponses.ts): The current keyword-based tutor implementation that the ML chatbot will replace.
5. [`src/pages/Practice.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Practice.tsx): The adaptive practice and evaluation center. Review how questions are answered, scored, and how post-test evaluation reports are displayed.
6. [`src/data/mockQuestions.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/data/mockQuestions.ts): The `PracticeQuestion` interface and sample questions. The ML question generator will output this exact structure.
7. [`src/pages/Assessment.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Assessment.tsx): The 6-step diagnostic onboarding flow that creates baseline user profiles.
8. [`src/data/mockLearningPath.ts`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/data/mockLearningPath.ts): Defines the `PathNode` structure and learning roadmap milestones.
9. [`src/pages/LearningPath.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/LearningPath.tsx): Roadmap UI showing milestone statuses (`completed`, `current`, `recommended`, `adapted`, `locked`).
10. [`src/pages/Learn.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Learn.tsx): The lesson player with integrated side tutor chat, "Explain simpler", and "Give me an example" interactive modes.
11. [`src/pages/DashBoard.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/DashBoard.tsx): The main dashboard containing the real-time "Your Tutor Adapted Your Path" alert banner.
12. [`src/pages/Progress.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/pages/Progress.tsx): The skill radar chart and continuous assessment analytics showing proof of measurable improvement.
13. [`src/App.tsx`](file:///c:/Users/aksha/Desktop/build%20ai/ML-Learner/client/src/App.tsx): Routing configuration showing how pages link together.

---

## 19. DIVISION OF RESPONSIBILITIES

```text
┌──────────────────────────────────────┬──────────────────────────────────────┐
│       FRONTEND RESPONSIBILITIES      │       ML / BACKEND RESPONSIBILITIES  │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • UI rendering & KaTeX / Markdown    │ • Prompt engineering & LLM pipelines │
│ • User gesture handling & navigation │ • Dynamic question synthesis         │
│ • Local optimistic state updates     │ • Non-repetition & distractor logic  │
│ • Fallback to mock data on error     │ • Knowledge state estimation (BKT)   │
│ • Profile switcher toggle            │ • "Why this question" explanation gen│
│ • Practice test timing & selection   │ • Multi-turn conversation context    │
│ • Toast notifications & visual flair │ • Secure API key management in .env  │
├──────────────────────────────────────┴──────────────────────────────────────┤
│                            SHARED RESPONSIBILITIES                          │
│ • Strict adherence to JSON API contracts (/tutor/chat, /questions/generate) │
│ • Consistency in question schema (PracticeQuestion interface)                │
│ • Consistent threshold alignment (<60% = reduce, 60-84% = maintain, >=85%   │
│   = increase)                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*This specification serves as the comprehensive engineering blueprint for the ML/AI integration into the LearnAI platform.*


---

## 20. ML IMPLEMENTATION ADDENDUM (SOURCE OF TRUTH)

> This section records what was actually built and decided. **Where it conflicts with Sections 1–19, Section 20 wins.**

### 20.1 Runtime & Deployment
- FastAPI app, single uvicorn worker, deployed on the Render free tier (512 MB RAM, 0.1 CPU, ephemeral disk).
- Measured memory use is about 75 MB.
- Banned dependencies: torch, tensorflow, transformers, sentence-transformers, sklearn, pandas, numpy, langchain, llama-index, chromadb, faiss, and the openai and groq SDKs.
- Allowed dependencies: fastapi, uvicorn, pydantic(-settings), sqlalchemy 2, psycopg 3, httpx, python-dotenv.
- `DATABASE_URL` defaults to SQLite. `postgres://` and `postgresql://` URLs are rewritten to `postgresql+psycopg://` (Neon/Supabase). Tables are created and the three demo learners are seeded on startup.
- **LLM:** Groq's OpenAI-compatible REST API, called through `httpx` with no SDK.
  - Models come from `GROQ_MODEL_MAIN` (default `openai/gpt-oss-120b`) and `GROQ_MODEL_FAST` (default `openai/gpt-oss-20b`). `GROQ_REASONING_EFFORT` defaults to `low`.
  - The app starts without `GROQ_API_KEY`; LLM features then return `LLM_NOT_CONFIGURED` along with fallback content.
- `ENABLE_DOCS=true` exposes `/docs` and `/openapi.json`. It is off by default.

### 20.2 API: Single Endpoint (replaces §12)
- **Routes:** only `GET /health` (returns `{status, db, llm_configured, pool_size}`) and `POST /api/v1/learnai`.
- **Request:** `{action, learner_id, payload}`.
- **Response, always this shape:** `{success, action, data, error: {code, message, details} | null, fallback_data}`. Business errors return HTTP 200 with `success: false`. The server never returns a raw 500.
- **Actions:** `get_profile`, `reset_learner`, `evaluate`, `generate_questions` (Phase 3), `tutor_chat` (Phase 4), `get_path` and `assessment` (Phase 5). `NOT_IMPLEMENTED` is no longer used.
- **Error codes:**
  - `UNKNOWN_ACTION`, `LEARNER_NOT_FOUND`, `NOT_IMPLEMENTED`, `UNKNOWN_CONCEPT`, `CANNOT_GRADE`, `INVALID_PAYLOAD`, `INVALID_REQUEST`, `INTERNAL_ERROR`
  - LLM failures: `LLM_NOT_CONFIGURED`, `MODEL_RATE_LIMIT`, `MODEL_TIMEOUT`, `MODEL_ERROR`, `GENERATION_FAILED`, `VERIFICATION_SHORTFALL`
  - Tutor: `EMPTY_MESSAGE`, `MESSAGE_TOO_LONG`, `QUESTION_NOT_FOUND`
- Demo learner ids: `alex-beginner`, `akshat-intermediate`, `elena-advanced`.

### 20.3 Concepts
- 26 concepts across the 6 skill categories, each with an id, name, category, prerequisites and aliases. The prerequisites form a DAG, and cycles are checked at startup.
- **Resolution** (`resolve_concept`), in order:
  1. exact id
  2. case- and punctuation-insensitive name or alias
  3. token overlap (Dice score ≥ 0.5, stop words removed, light plural stripping), with `category_hint` breaking ties
  4. strict fuzzy match for typos (difflib ≥ 0.9)
- If nothing matches, the response is `UNKNOWN_CONCEPT` with the 3 closest suggestions.

### 20.4 Learner Model (replaces the §8.3 EMA)
- **Mastery** is tracked per concept on a 0–100 scale.
  - **Correct answer:** Easy +3, Medium +5 or Hard +8, multiplied by `(1 − m/100)·1.5`. The minimum gain is +1, and an answer under 20 s earns +1 more.
  - **Wrong answer:** Easy −8, Medium −6 or Hard −3, multiplied by `(m/100)·1.5`. The minimum loss is −1.
  - The result is clamped to 0–100, and the last 10 results are kept.
- **Confidence:** `min(1, attempts/10) · (1 − 0.15·min(consecutive_wrong, 3))`.
- **Trend:** accuracy of the last 3 results compared with the 3 before them. A difference above +0.15 is `improving` and below −0.15 is `declining`. With fewer than 4 results, the trend is `stable`.
- **Category skill:** attempt-weighted mean of concept masteries. Concepts with no attempts have weight 1.
- **`accuracyRate`:** the §8.3 cumulative formula, stored to one decimal place.

### 20.5 Weakness Classification
- **`weakness_report`:**
  - Strong is 75 or above, Average is 60–74, and Weak is below 60.
  - **Relative rule:** a concept below 75 is also Weak if it sits at least 15 points under the learner's average practiced mastery. This catches Elena's 68% on RAG chunking.
  - Untried concepts (0 attempts) are never classified as weak; they are listed under `untried`.
- **`profile.weaknesses`:** keeps the §5 cut-off of 65, plus the relative rule, for frontend compatibility.
- **Priority** (0–100): `0.45·(100−mastery) + 0.20·min(consecutive_wrong,3)/3·100 + 0.15·(100−recent_accuracy·100) + 0.10·(100 if declining, 50 if stable, 0 if improving) + 0.10·min(attempts,10)/10·100`.
  - HIGH is 55 or above (lowered from 60 in Phase 3), MEDIUM is 40–54, and LOW is below 40.
  - Only HIGH entries get a `tutor_alert`.
- **`recommended_revision`:** walk the prerequisites one layer at a time, starting with the direct ones. In the closest layer that has any concept below 60, pick the weakest. If no prerequisite has a gap, recommend the concept itself.

### 20.6 Adaptive Difficulty
The rules look at the last 5 results and run in order; the first match wins.

| Rule | Condition | Next difficulty |
|---|---|---|
| R0 | No attempts yet | Easy if mastery is below 50, Medium if below 75, otherwise Hard |
| R1 | Last two answers wrong at Hard | Medium |
| R2 | Last two answers wrong at Medium | Easy |
| R3 | 2 or more of the last 3 wrong | One level down |
| R4 | Last 3 correct at the same level, with average time under 60% of expected (Easy 30 s, Medium 60 s, Hard 120 s) | One level up |
| R5 | Last 3 correct at the same level | Up only if mastery is at least 60 (Easy → Medium) or 75 (Medium → Hard); otherwise stay |
| R6 | Last answer correct but slower than 1.5× expected | Stay |
| R7 | Anything else | Stay |

### 20.7 Adaptation
- **Action from the score** (§8.1 thresholds): below 60 is `reduced`, 60–84 is `maintained`, and 85 or above is `increased`.
- **Override:** when the difficulty engine disagrees, its decision wins and the `reason` explains why. There is no override when the engine simply cannot move: already at Easy for `reduced`, or at Hard for `increased`.
- **Roadmap fast-track:** only when the score is 85 or above and the concept is not Weak. A step-up during recovery leaves the roadmap unchanged.
- **Event fields:** `reason`, `recommendation` and `pathAdjustment` are generated from facts in Python.
- **Storage and shape:** events are saved to `adaptive_events` and returned in the §8.2 shape, plus `createdAt`, `concept`, `overridden` and `thresholdAction`. `get_profile` returns the last 5 in `profile.recentAdaptiveEvents`.

### 20.8 Evaluate
- **Payload:** a single answer, or a quiz: `{quiz: true, topic, category, answers: [...]}`.
- **Grading order:**
  1. a question stored server-side (its `correct_index` is authoritative)
  2. the payload's `correct_index`
  3. the payload's `is_correct`
  4. otherwise `CANNOT_GRADE`
- **Transactions:** one transaction per call.

### 20.9 Question Generation (Phase 3)
- **Pipeline:** Python makes every factual decision and the LLM only writes the question content.
- **Concept choice:**
  - If `target_concept` is given, it is resolved and used.
  - Otherwise, within the requested category, pick the highest-priority Weak concept. If there is none, pick the lowest-mastery practiced concept, then the first concept whose prerequisites are all at 60 or above.
  - With no category, the same order applies across the whole catalog.
- **Difficulty:** `next_difficulty` when the request says `Adaptive`, otherwise the requested level.
- **Variety:** a scenario domain (10 options) and a question angle (6 options) are assigned to each question.
  - Values not used recently for the same concept come first, otherwise the least recently used. The window is the learner's last 20 questions.
  - Beginners never get `code_debugging`, and their `formula_interpretation` questions use plain English.
  - Recent titles are sent to the model as "do not repeat". A question whose normalized text has a difflib ratio above 0.85 against a recent one is rejected.
- **One LLM call per request:** `GROQ_MODEL_MAIN`, JSON mode, temperature 0.6. Level calibration follows §6 and §14.2.
- **Validation:**
  - Exactly 4 distinct, non-empty options; `correct_index` between 0 and 3; non-empty explanation and hint.
  - The hint must not contain the correct option's text.
  - Options must not be referred to by letter or position, because they are shuffled.
  - Code is at most 15 lines. Code pasted into the question text is moved to `code_snippet`.
  - **Repair retry:** one retry for the failed slots only, with the errors included in the prompt. Valid questions are kept even if the retry fails.
- **Post-processing:** options are shuffled with a seeded random. Ids are `gen-<uuid>`. Difficulty, category and concept are set by Python.
  - `whyThisQuestion` is built only from real facts, never by the LLM.
  - `recommendedNextDifficulty` is the difficulty engine's output.
  - Every question, including `correctIndex`, is saved to `questions`, so `evaluate` grades it on the server.
- **Fallback:** `app/data/fallback_questions.json` holds 12 curated MCQs, 2 per category.
  - Used on an `LLMError` or when no question passes validation. The response then has `success: false` with the error code, and the questions in `fallback_data.questions`, preferring the same concept or category.
  - Fallback questions are also saved, so they can be graded.
- **Groq client:**
  - One shared `httpx.AsyncClient`, opened and closed with the app.
  - Uses `message.content`; any `reasoning` field is ignored, and empty content is an error.
  - Sends `reasoning_effort` when it is set. If Groq rejects it with a 400, the request is retried once without it.
  - **Retries:** one retry on 429, 5xx or a timeout.
  - **Error codes:** `MODEL_RATE_LIMIT`, `MODEL_TIMEOUT`, `MODEL_ERROR`, `LLM_NOT_CONFIGURED`. A missing key is detected without making a request.
  - **Logging:** model, latency and token counts only, never prompts or completions.
- **Weakness priority labels:** HIGH is now 55 or above (was 60), MEDIUM is 40–54, LOW is below 40.

### 20.10 Question Correctness and Serving Chain (Phase 3.5)
- **Generator** (`app/prompts/question_generator.txt`, version `qgen-v2`, temperature 0.5):
  - Exactly one option must be correct under every reasonable interpretation, and every distractor needs a statable reason.
  - Numeric questions must be internally consistent and the explanation must show the computation.
  - "Best action" questions must not list two valid remedies.
  - Code must be mentally executed; code goes only in `code_snippet`.
  - Scenarios must be realistic, and formulas or metrics must not be invented.
  - Output includes `distractor_reasons` (exactly 3). These are used for validation only and never returned.
- **Python validation (on top of Phase 3):**
  - `distractor_reasons` must have 3 entries; when the model sends 4, the entry for the correct option is dropped.
  - A question may not refer to "the following code" without `code_snippet`.
  - An explanation that admits another option "would also help" is rejected as a second correct answer.
- **Verifier** (`app/llm/verifier.py`):
  - **One call per batch:** `GROQ_MODEL_FAST`, JSON mode, temperature 0, reasoning effort medium.
  - **Input:** `{qid, question, code_snippet, options}` only. It never sees the answer, explanation, hint or distractor reasons.
  - **Output:** `{answer_index, confidence, multiple_correct, flawed, issue}`.
  - **Rejection:** if the answer differs from ours, `multiple_correct` is set, `flawed` is set, or confidence is below 0.6.
  - **Regeneration:** only rejected or invalid questions are regenerated, once (one more generate call and one more verify call).
  - **When the verifier fails:** with `VERIFY_QUESTIONS=true` (the default), no unverified question is ever served. A verifier failure (rate limit or timeout) goes to the pool and fallback chain.
- **Serving chain:**
  1. LLM generation plus verification.
  2. **Pool:** questions with `verified=true` and `source="llm"` at the same level, never already served to this learner. The relaxation order is same concept and difficulty, then same concept with an adjacent difficulty, then same category and difficulty. The least-served questions come first.
  3. The static curated bank.
  - Each question carries `source` (`llm` | `pool` | `fallback`) and `verified`.
  - If the requested count is met, the response is `success: true` with a notice in `error: {code, message}` whenever the pool or bank was used.
  - `data.generation` reports `verified`, `rejected` and `rejection_issues` (at most 10). `data.sources` counts questions by source.
- **Storage:**
  - `questions` gains `verified`, `level`, `source` and `times_served`. These columns are added in place on startup, without migrations.
  - New table `question_serves (learner_id, question_id, served_at)`, unique per pair.
  - `reset_learner` clears that learner's serves and chat messages.
- **Groq client:**
  - **429:** wait and retry once only if `Retry-After` is 3 s or less; otherwise fail fast with `MODEL_RATE_LIMIT`.
  - **Concurrency:** at most 2 Groq calls at once (`asyncio.Semaphore`).
  - **Reasoning headroom:** `max_tokens` covers the visible answer. The client adds headroom for gpt-oss reasoning tokens (low 512, medium 1536).
  - **JSON salvage:** when Groq returns `json_validate_failed`, the client tries to salvage `failed_generation`, for example by repairing raw LaTeX backslashes.

### 20.11 Tutor Chat (Phase 4)
- **Payload:** `{message (1-2000 chars), conversation_id?, mode?, current_topic?, active_lesson_id?, question_id? (required for hint), student_answer? (required for evaluate), record? (default true)}`.
- **Mode auto-detection** (`app/engine/tutor_modes.py`, first match wins):
  1. hint (needs `question_id`)
  2. simplify (includes Hinglish "samajh nahi" and "aasan")
  3. code
  4. example ("udaharan")
  5. practice ("question do")
  6. revision
  7. path ("aage kya")
  8. evaluate (when `student_answer` is present)
  9. explanation (the default)
- **Mode settings:**

  | Mode | Temperature | max_tokens |
  |---|---|---|
  | explanation | 0.5 | 900 |
  | simplify | 0.6 | 600 |
  | example | 0.7 | 800 |
  | code | 0.2 | 1000 |
  | practice | 0.4 | 700 |
  | hint | 0.3 | 300 |
  | evaluate | 0.1 | 700 |
  | revision | 0.3 | 800 |
  | path | 0.3 | 500 |

  - Repair calls use temperature 0.2. All calls use `GROQ_MODEL_MAIN`.
- **Concept for the turn, in order:**
  1. `current_topic`
  2. `active_lesson_id`
  3. the stored question's concept (hint and evaluate modes)
  4. a concept phrase in the message (skipped when the message is flagged as an injection attempt)
  5. the last concept in the conversation
  6. none
- **Context** (`app/engine/tutor_context.py`, at most 700 estimated tokens):
  - The learner's name, level, goal, pace, languages and up to 10 known topics.
  - The current concept: mastery, trend, confidence, status and priority, plus its prerequisites with their mastery and the weakest one.
  - The top 3 weaknesses, top strengths, and the last adaptive event.
  - In hint and evaluate modes, the stored question.
  - **History:** the last 6 messages, each cut to 600 characters, sent as chat turns. Learner turns are wrapped in `<learner_message>`.
- **Prompts** (`app/prompts/*.txt`, version `tutor-v1`): loaded once at startup and rendered with `format_map` and safe defaults.
  - **System prompt order:** role, guardrails, learner context, the rules for the learner's level only, the rules for the active mode only, then the output schema.
  - **Version:** returned in every response and stored on every chat message.
- **Output schema (LLM):** `{message, concept, follow_up_suggestions[3], checkpoint_question|null}`, plus `evaluation` in evaluate mode.
- **Added by Python, never by the LLM:** `difficultyLevel`, `recommendedNextAction {type: practice|revise|advance|continue, targetTopic, reason}`, mode, `conversationId`, `promptVersion`, `guardrails`, `evaluation`, `masteryUpdate` and `adaptiveDecision`.
  - The tutor response uses camelCase keys.
- **Guardrails:**
  - **Input:** control characters are stripped; errors are `EMPTY_MESSAGE` and `MESSAGE_TOO_LONG`.
  - **Pre-classifier** (pure Python): flags injection attempts, distress (English and Hinglish), clearly off-topic messages, and Hinglish input. A Hinglish message makes the prompt require a Hinglish reply.
    - An off-topic message with no ML terms gets a template reply without calling the LLM.
  - **Output checks:**
    - Schema validation.
    - System-prompt leakage: 4 sentinel phrases that appear in the prompt.
    - Hint leakage: answer phrases, the correct option's text, or a difflib ratio above 0.8 on any sentence.
    - Beginner math: a math line with no plain-English sentence right after it is removed.
    - Checkpoint text, options or answers, and follow-up lists, written into `message` are cut, because those have their own fields. A live eval caught a practice reply that wrote "Correct index: 2" into its message.
    - Length cap by level (Beginner 250, Intermediate 400, Advanced 550 words), adjusted for pace with 1.2× tolerance, and capped further by mode (simplify 200, practice 150, hint 90, path 250). Trims never cut inside a code block.
    - Links are removed unless they point to docs.python.org, scikit-learn.org, pytorch.org, numpy.org or pandas.pydata.org.
    - Distress turns: the reply always ends with `TUTOR_SUPPORT_TEXT`, there is no checkpoint, the follow-ups do not push studying, and the prompt forbids teaching in that turn.
  - **Repairs:** leakage and schema violations get one repair call. If that still fails, a safe fallback reply is returned (for hints, the stored question's curated hint).
  - **Logging:** flags, mode, concept, latency and token counts only, never message contents.
- **Checkpoint questions:**
  - Only in explanation, simplify, example and practice modes; required in practice.
  - Validated with the question schema and checked by the verifier.
  - Practice mode gets one regeneration. If it still fails, the question is dropped and the reply says so.
  - Valid checkpoints are saved to `questions` (`source=llm`, `verified=true`, id `chk-…`), so `evaluate` grades them on the server.
- **Evaluate mode:** when a concept is resolved and `record` is not false, the turn records an attempt through the Phase 2 loop (Medium difficulty, correct if the score is 60 or above). It returns `masteryUpdate` and `adaptiveDecision`.
- **LLM failure:** `success: false`, with `fallback_data {conversation_id, reply (the concept's 2-line summary or the distress reply), follow_ups (3)}`. The user message is still saved.

### 20.12 Phase 4 Follow-up Fixes (Phase 5)
- **Evaluate-mode score:** when `tutor_chat` evaluate mode records an attempt, the AdaptiveEvent score and the adaptation decision use the real graded score (for example 40). Mastery still uses the binary result (correct when the score is 60 or above), and `accuracyRate` still counts answers.
- **Verifier difficulty:** the verifier also returns `perceived_difficulty` (Easy, Medium or Hard).
  - `generate_questions` rejects a question rated two levels away from the requested difficulty (Easy vs Hard).
  - Tutor checkpoints reject an Easy question for an Advanced learner and a Hard question for a Beginner.
- **KaTeX balance:** `$…$`, `$$…$$`, `\[…\]` and `\(…\)` are checked per prose paragraph; code blocks and inline code are skipped, so math can never be counted across a code block.
  - An unclosed opener is closed at the end of its paragraph.
  - A closer with no opener cannot be repaired deterministically, so it triggers the repair call.
  - **Known limitation:** a lone currency `$` gets a closing `$`.

### 20.13 Recommendation Engine and Learning Path (Phase 5)
- **Goals** (`app/engine/goals.py`): four goal keys, ML Engineer, Interviews, GenAI Apps and ML from Scratch, each mapping to target concepts.
  - Free-text goals are resolved by keyword aliases, then a difflib match. The default is ML Engineer.
  - The learner's `target_topics` are always added to the targets; a category name such as "Deep Learning" adds every concept in that category.
- **Path algorithm** (`app/engine/recommender.py`, pure functions):
  - **Closure:** the targets plus all their DAG ancestors.
  - **Classification:**
    - A practiced Weak concept becomes a revision node.
    - A concept at mastery 75 or above is completed (`skipped=true` when it was never practiced).
    - Everything else is still to learn.
  - **Ordering:** Kahn's topological sort with a heap. Among available nodes, priority goes to:
    1. revision nodes, highest weakness priority first
    2. unfinished foundations of weak concepts ("fix the foundation first")
    3. goal relevance: shortest DAG distance to a target
    4. category order
    5. concept id
  - **Statuses:**
    - completed nodes first
    - the first unfinished node is `current`
    - revision and foundation nodes are `adapted`
    - the next 2 are `recommended`
    - later nodes are `locked` while any prerequisite is unfinished, otherwise `recommended`
  - **Estimates:** `estimatedMinutes` = category base (Python 30, Statistics 40, ML 50, DL/NLP/GenAI 60) × (1 − mastery/100) × pace factor (Relaxed 1.2, Balanced 1.0, Intensive 0.85), minimum 10. Completed nodes count 0.
    - **Weeks remaining:** total minutes ÷ (daily minutes × 6), rounded up.
- **PathNode:** `{id, conceptId, title, category, status, progress, mastery, estimatedMinutes, reason, isRevision, skipped, prerequisites, foundationFor, order}`. `foundationFor` is the weak concept a foundation node was moved earlier for, or null.
  - Milestones are one stage per category in path order: `{id, title, category, status: completed|current|upcoming, nodeIds, progress}`.
- **Snapshots** (table `learning_paths`): a row is written only when the path's structure hash (order, status, revision, skipped) changes.
  - `diff` reports `inserted_revision`, `skipped`, `completed`, `unlocked` and `reordered` changes, plus a one-line `beforeAfter`. With no change it reads "Roadmap unchanged: current focus remains X."
  - **Evaluate:** each call compares the path before and after the answer, snapshots it, and writes that real summary into the AdaptiveEvent's `pathAdjustment`.
  - **Tutor path mode:** the context includes the current node, the next 2 nodes and the latest change.
- **whyThisPath:**
  - Python builds the facts: goal, skipped concepts with mastery, revisions, foundations, current and next nodes, pace, daily minutes and weeks remaining.
  - `GROQ_MODEL_FAST` (temperature 0.3, at most 90 words, prompt `app/prompts/why_this_path.txt`) rewrites them.
  - **Validation:** every number in the output, including spelled-out numbers ("three weeks"), must appear in the facts; otherwise the Python template is used.
  - **Caching:** the text is stored on the snapshot, so `get_path` makes no new LLM call while the path hash is unchanged. An LLM failure falls back to the template, and `success` stays true.
- **`get_path` payload:** `{goal?}`. A goal here is a "what if" preview: it is computed but never saved.
- **`get_path` response:** `{goal, goalLabel, targets, nodes, milestones, currentNode, nextNodes, changes, changedNow, beforeAfter, whyThisPath, whySource, estimatedWeeksRemaining, totalEstimatedMinutes, dailyPlan, pathHash, preview}`.
- **dailyPlan:** at most 1 revision (the top weakness, unless it is the current node), then practice on the current node at the engine's next difficulty, then a lesson on the current node.
  - The lesson is capped at the node's own estimate.
  - The total always fits within `dailyCommitmentMinutes`.

### 20.14 Assessment, Pool Warm-up and Health (Phase 5)
- **`assessment` payload:** `{name?, experience_level, languages, topics_known, goal, pace, daily_minutes, overwrite?}`.
  - The `learner_id` comes from the request.
  - An existing learner needs `overwrite=true`, otherwise the response is `LEARNER_EXISTS`.
  - Demo and pool learners cannot be assessed: `DEMO_LEARNER_PROTECTED`.
  - `learner_id` must match `^[a-z0-9][a-z0-9_-]{2,63}$`.
- **Level** (§7.1 rule): Beginner if experience is Beginner or 1 or fewer topics are known; Advanced if experience is Advanced or 6 or more topics are known; otherwise Intermediate.
- **Baselines** (`app/engine/cold_start.py`; every concept starts with 0 attempts, confidence 0.1, trend stable):
  1. Experience default (Beginner 15, Some Programming 25, Intermediate 40, Advanced 55), applied in topological order and capped by the concept's prerequisites.
  2. Python known → Python concepts 80 (Python Basics 85). No language → Python concepts 15.
  3. Known topics → 78, and every DAG ancestor at least 70. Broad topics map to core concepts, for example Statistics → probability, std/variance and distributions.
  4. §7.1 gaps: if Statistics isn't known, statistics concepts are capped at 50; if ML isn't known, Bias vs Variance is capped at 55. The gaps never override knowledge implied by rule 3.
- **Response:** `{profile, weakness_report, path, level, goal, firstStep: {type: "diagnostic", conceptIds (3 unfinished path concepts nearest 50 mastery), message}}`.
  - The first path snapshot is saved.
  - The only LLM call is the whyThisPath polish.
- **`reset_learner`:**
  - Demo learners: as before, and it also clears path snapshots.
  - Assessed learners: `NOT_A_DEMO_LEARNER`.
  - Unknown ids: `LEARNER_NOT_FOUND`.
- **Pool learners:** `pool-beginner`, `pool-intermediate` and `pool-advanced` are created on startup.
  - Every action except `generate_questions` and `assessment` treats them as not found.
  - `scripts/warm_pool.py` uses them to pre-generate verified questions per cell (concept × level × difficulty).
  - **Cells skipped:** Hard for Beginners, Easy for Advanced learners, and Hard for foundational Python.
  - **Resumable:** cells that already have N verified questions are skipped.
- **Health:** `GET /health` returns `{status, db: ok|error, llm_configured, pool_size}`. It runs one trivial query and one count, and never calls the LLM.
- **Error codes added:** `LEARNER_EXISTS`, `DEMO_LEARNER_PROTECTED`, `NOT_A_DEMO_LEARNER`.

### 20.15 Deployment Readiness (Phase 6)
- **Recommender:** for learners who know no programming language (empty, or only "None"), category order (rule d) outranks goal relevance (rule c), so the Python foundations come first. Everyone else keeps the original order.
- **KaTeX check:** a dollar sign directly followed by an amount (`$50`, `$1,200.50`) is treated as currency, not math. This applies both to the balance check and to the Beginner math rule.
- **Docs:** `ENABLE_DOCS` defaults to true.
  - `/docs` (Swagger) and `/redoc` are on in production.
  - The app description is the full frontend integration guide (`app/api_docs.py`).
  - `POST /api/v1/learnai` carries 16 runnable request examples covering every action and tutor mode.
  - Response schemas (`app/api_models.py`) are documentation-only: `response_model=None`, so the returned envelope is unchanged.
- **Seed question:** `seed-q-bias-variance` (a curated Bias vs Variance question, `source="fallback"`, kept out of the pool) is created on startup, so the hint and evaluate examples work on a fresh database.
- **Abuse protection** (`app/protection.py`):
  - Per-IP sliding 60-second window. The client IP is the first `X-Forwarded-For` hop.
  - **Limits:** AI actions (`tutor_chat`, `generate_questions`, `assessment`) 20 per minute; other actions 120 per minute. Configure with `RATE_LIMIT_LLM_PER_MIN` and `RATE_LIMIT_OTHER_PER_MIN`; 0 disables a limit.
  - **When exceeded:** HTTP 200, `success: false`, `RATE_LIMITED`, with `details.retry_after_seconds`.
  - **Memory:** bounded to 5000 tracked keys, least-recently-used evicted.
  - **Body size:** request bodies over 32 KB get HTTP 413 `PAYLOAD_TOO_LARGE` in the standard envelope.
- **CORS:** `ALLOWED_ORIGINS="*"` allows any origin (no credentials).
- **Postgres (Neon):**
  - The engine uses `pool_pre_ping`, `pool_size=3`, `max_overflow=2` and `pool_recycle=300`.
  - Connections use `connect_timeout=20` and `prepare_threshold=None` (safe behind PgBouncer).
  - A connection that fails while Neon wakes up is retried (3 attempts, 1.5 s backoff).
  - **Tests:** with `LEARNAI_TEST_PG=1` the suite runs in a throwaway `test_<random>` schema on Neon's direct host. It checks `current_schema()` before any write and drops the schema at exit.
- **Startup:** idempotent `create_all` plus additive column migration, then seeding only if empty. It logs version, DB type, `llm_configured`, `pool_size` and prompt versions, never secrets.
- **Health:** `GET /health` runs one indexed count, which doubles as the DB check.
