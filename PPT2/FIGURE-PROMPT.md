## FIGURE-FIRST PRESENTATION REQUIREMENT

The review panel responds well to **figures, diagrams, architecture diagrams, flowcharts, plots, tables, and visual comparisons**.

Therefore, do NOT produce a text-heavy presentation.

I want you to deliberately design the **visual content strategy** for every slide.

For every slide, tell me exactly:

1. **What figure/visual should be on the slide**
2. **Whether the figure already exists in the uploaded files**
3. **Which exact existing figure/table/plot I should reuse**, if available
4. **If no suitable figure exists, what new figure I should create**
5. **Exactly what the new figure should contain**
6. **What labels/nodes/arrows/components should appear**
7. **Why that figure is useful for the slide**
8. **Whether the figure should occupy approximately 1/3, 1/2, or 2/3 of the slide**

Do not simply write:

> "Add an architecture diagram."

Instead write something like:

> **Figure:** Phase 1 fault-localization pipeline  
> **Source:** Phase 1 paper, Figure X  
> **Use:** Reuse this figure  
> **Placement:** Right 55% of slide  
> **Purpose:** Shows the complete Phase 1 workflow from grid data → graph construction → temporal windows → GNN → fault localization.  
> **Do not duplicate:** Remove detailed implementation labels if they make the figure unreadable at PPT scale.

---

# 1. CREATE A FIGURE INVENTORY FIRST

Before creating the slide content, inspect all uploaded files and create an inventory of available visual material.

Look specifically for:

- Architecture diagrams
- System diagrams
- Workflow diagrams
- Power-grid topology figures
- IEEE test-system diagrams
- Graph representations
- Data-generation diagrams
- Dataset figures
- Fault-location illustrations
- Model architecture diagrams
- Training curves
- Validation curves
- Loss curves
- Confusion matrices
- ROC curves
- Precision-recall curves
- F1 comparisons
- Model comparison tables
- Ablation studies
- Hyperparameter tables
- Experimental result tables
- Screenshots of implementation
- Code-generated plots
- Any figure already used in the Phase 1 paper
- Any figure present in the Phase 2 MD documentation

Create a table:

| Figure | Source | Phase | What it shows | Reuse? | Recommended slide |
|---|---|---|---|---|---|
| Figure 1 | Phase 1 paper | Phase 1 | ... | Yes | Slide X |
| Figure 2 | ... | ... | ... | ... | ... |

Do not invent figure numbers. Use the actual figure/table names or numbers from the uploaded files.

---

# 2. PRIORITIZE EXISTING RESEARCH FIGURES

If the Phase 1 paper already contains a strong figure, prefer reusing it rather than creating an unnecessary new figure.

Especially look for:

### A. Power-grid/test-system figure

Use it to visually introduce the problem.

The audience should see:

**Power grid → buses/nodes → transmission/distribution connections → fault location**

If the paper has an appropriate grid topology figure, reuse it.

---

### B. Phase 1 methodology/pipeline

Find the strongest Phase 1 workflow diagram.

It should communicate something like:

**Power-grid data**

↓

**Graph construction**

↓

**Node/edge features**

↓

**Temporal windows**

↓

**GNN/STGNN**

↓

**Fault detection/localization**

Use the actual pipeline from my paper.

---

### C. Phase 1 architecture

If there is a GCN/GraphSAGE/GAT/GRU/STGNN architecture figure, use it when explaining the Phase 1 model.

Do not overload the slide with multiple architectures.

Choose the figure that best explains the main Phase 1 approach.

---

### D. Phase 1 results

If there are actual experimental plots/tables, prioritize visual presentation.

For example:

- Model comparison bar chart
- F1 comparison
- Precision/Recall comparison
- ROC/AUC
- Localization accuracy
- Confusion matrix

Use the actual results.

If the results are in a table but a bar chart would communicate them better, recommend converting the table into a PPT chart.

Never alter the underlying numbers.

---

# 3. PHASE 1 LIMITATIONS SHOULD ALSO BE VISUAL

Do NOT make the Phase 1 limitations slide a wall of bullets.

Create a visual cause-effect diagram.

For example:

**Phase 1 Approach**

↓  

**Observed issue**

↓  

**Why it matters**

↓  

**Requirement for Phase 2**

The exact content must be derived from my research.

For example, if class imbalance was genuinely one of the problems:

**Highly imbalanced fault labels**

→

**Minority fault samples**

→

**Difficult learning/localization**

→

**Need for improved learning strategy**

But do NOT assume class imbalance is the main limitation unless my files show this.

---

# 4. MAKE THE PHASE 1 → PHASE 2 TRANSITION VISUAL

This should be one of the best-looking slides.

Create a visual timeline/process:

### PHASE 1
**Groundwork**

Research problem  
↓  
Dataset & graph formulation  
↓  
Baseline/STGNN models  
↓  
Experiments  
↓  
Results  

### LIMITATIONS IDENTIFIED

↓  

### PHASE 2
**PCSTGNN**

New design motivation  
↓  
New architecture  
↓  
Implementation  
↓  
Training  
↓  
Evaluation

Use arrows to make the research progression obvious.

The viewer should understand the evolution within 5 seconds.

---

# 5. "WHY PCSTGNN?" SHOULD HAVE A VISUAL

Create a **problem → solution mapping figure**.

Recommended format:

| Phase 1 observation | Phase 2 design response |
|---|---|
| Actual limitation | PCSTGNN component |
| Actual limitation | PCSTGNN component |
| Actual limitation | PCSTGNN component |

Alternatively, create connected blocks:

**Problem 1** → **PCSTGNN component 1**

**Problem 2** → **PCSTGNN component 2**

**Problem 3** → **PCSTGNN component 3**

This should be based strictly on the actual architecture.

---

# 6. PCSTGNN ARCHITECTURE MUST HAVE A LARGE FIGURE

This is probably the **most important figure in the Phase 2 section**.

Create a dedicated architecture slide.

The architecture diagram should show the actual information flow.

For example, depending on the implementation:

**Power-grid measurements**

↓

**Graph construction**

↓

**Node / edge features**

↓

**Spatial processing**

↓

**Temporal processing**

↓

**Physics-informed component**

↓

**Fault representation**

↓

**Fault localization output**

But do NOT assume these exact blocks.

Read the PCSTGNN implementation and construct the diagram from the actual architecture.

Every box in the diagram should correspond to a real component.

Every arrow should represent a real data flow.

---

# 7. CREATE A SECOND PCSTGNN FIGURE IF NECESSARY

If the architecture is sufficiently complex, use two figures rather than making one unreadable figure.

### Figure 1 — High-level PCSTGNN

Show only:

Input → Spatial module → Temporal module → Physics module → Output

### Figure 2 — Detailed PCSTGNN

Show:

- Feature dimensions
- Graph operations
- Temporal sequence
- Physics-informed mechanism
- Loss/output heads

Only include details that are genuinely present in the implementation.

---

# 8. SHOW THE POWER-GRID + GRAPH CONNECTION

A particularly useful figure would show:

### Physical power grid

**Bus 1 — Bus 2 — Bus 3 — Bus 4**

↓

### Graph representation

**Node 1 — Node 2 — Node 3 — Node 4**

with:

- Node features
- Edge connectivity
- Temporal measurements

↓

### GNN processing

This helps a panel understand **why a GNN is being used for a power grid**.

If the Phase 1 paper already has such a figure, reuse it.

Otherwise recommend creating it.

---

# 9. SHOW TEMPORAL MODELING VISUALLY

If temporal modeling is important to the project, include a simple figure such as:

**t−11 → t−10 → ... → t−1 → t**

Each timestep contains the graph:

**G₁ → G₂ → ... → G₁₂**

Then:

**Spatial representation + temporal evolution → fault localization**

This is much easier for the panel to understand than explaining temporal windows only with text.

Use the actual window size from the implementation.

---

# 10. SHOW PHYSICS-INFORMED COMPONENT VISUALLY

If PCSTGNN genuinely includes physics-informed constraints, create a visual showing:

**Grid measurements**

+

**Physical relationships / constraints**

↓

**PCSTGNN**

↓

**Fault localization**

Explain exactly what physical information enters the model.

Do not use the phrase "physics-informed" merely as a label.

Show what makes the architecture physics-informed.

If the architecture does NOT actually contain an explicit physics-informed mechanism, do not create this figure.

---

# 11. DATASET FIGURE

Include a visual representation of the dataset.

Depending on the actual dataset, show:

- Test system topology
- Number of buses/nodes
- Number of scenarios
- Number of samples/windows
- Number of features
- Fault classes
- Observation nodes
- Train/validation/test split

Prefer a diagram + small statistics panel rather than a paragraph.

Use exact numbers from the files.

---

# 12. EXPERIMENTAL PIPELINE FIGURE

Create a figure for the experimental methodology:

**Dataset**

↓

**Preprocessing**

↓

**Graph construction**

↓

**Temporal window generation**

↓

**Train / Validation / Test**

↓

**PCSTGNN**

↓

**Prediction**

↓

**Evaluation**

This can be one of the main methodology slides.

---

# 13. RESULTS SHOULD BE VISUAL

If genuine Phase 2 results exist, prioritize:

### First choice:
A model-comparison chart.

Example:

**F1 / Precision / Recall / AUC**

Phase 1 model  
Phase 2 PCSTGNN  
Other baselines

Use only real values.

### Second choice:
Training curves.

For example:

- Training loss
- Validation loss
- F1 over epochs

Only if the curves actually exist.

### Third choice:
Confusion matrix.

Only if classification results exist.

### Fourth choice:
Fault localization visualization.

If the project has a spatial grid and predicted fault locations, this could be particularly powerful:

**Actual fault location vs predicted fault location on the grid**

Only use this if such predictions actually exist.

---

# 14. IF PHASE 2 RESULTS DO NOT EXIST

Do NOT create fake graphs.

Instead, make the implementation status visual.

For example:

### Phase 2 Progress

**Dataset** ✓

**Preprocessing** ✓

**PCSTGNN implementation** ✓

**Training** → In progress

**Evaluation** → Pending

**Final comparison** → Pending

Use a clean progress/timeline visual.

This is academically much better than creating empty or fabricated result charts.

---

# 15. FIGURE PRIORITY

Rank the figures you recommend.

### MUST HAVE

Figures that are essential to understanding the research:

1. Power-grid/problem representation
2. Phase 1 methodology
3. Phase 1 model/architecture
4. Phase 1 results
5. Phase 1 limitations
6. Phase 1 → Phase 2 transition
7. Why PCSTGNN?
8. PCSTGNN architecture
9. Phase 2 experimental pipeline
10. Phase 2 results/status

### GOOD TO HAVE

11. Dataset visualization
12. Graph representation
13. Temporal-window visualization
14. Physics-informed mechanism
15. Phase 1 vs Phase 2 comparison
16. Detailed PCSTGNN internal architecture

### AVOID

Do not recommend:

- Stock images of power plants
- Generic AI/GNN pictures
- Decorative icons
- Generic neural-network illustrations
- Unrelated power-grid photographs

Every figure should contribute to understanding the research.

---

# 16. FIGURE DENSITY

The presentation should be visually rich but not overcrowded.

Aim for:

**~40% text + ~60% visual content**

where appropriate.

For architecture/methodology slides:

**~25–35% text + ~65–75% figure**

For result slides:

**~20% explanation + ~80% chart/table/plot**

For introduction slides:

**~50% text + ~50% conceptual visual**

Do not force a figure onto a slide where it doesn't improve understanding.

---

# 17. FOR EVERY SLIDE, USE THIS EXACT FORMAT

### Slide X — [Title]

**Purpose:**  
...

**Slide text:**
- ...
- ...
- ...

**PRIMARY FIGURE:**  
[Exact figure/diagram to use]

**Figure source:**  
[Phase 1 paper / Phase 2 MD / create new]

**Figure status:**  
[Existing / Modify existing / Create new]

**What the figure should contain:**  
...

**Figure placement:**  
[Left 50% / Right 50% / Center 70% / Full width]

**Why this figure matters:**  
...

**Speaker explanation:**  
...

---

# 18. CREATE A FINAL FIGURE CHECKLIST

At the end, provide:

## Figures I MUST HAVE

List the essential figures in order.

## Figures I CAN REUSE FROM PHASE 1 PAPER

Give exact figure numbers/titles where possible.

## FIGURES I SHOULD CREATE FOR PHASE 2

List new diagrams that need to be created.

## FIGURES THAT SHOULD BE GENERATED FROM EXPERIMENTAL RESULTS

List graphs that should be generated ONLY if the corresponding results exist.

## FIGURES I SHOULD NOT USE

List unnecessary/decorative figures that would make the presentation weaker.

---

# FINAL VISUAL PRINCIPLE

The presentation should visually tell this story:

**POWER GRID**

↓  

**FAULT LOCALIZATION PROBLEM**

↓

**PHASE 1 GROUNDWORK**

↓

**PHASE 1 EXPERIMENTS**

↓

**PHASE 1 RESULTS**

↓

**WHAT DID NOT WORK / WHAT REMAINED LIMITED**

↓

**WHY A NEW APPROACH WAS REQUIRED**

↓

**PCSTGNN**

↓

**HOW PCSTGNN ADDRESSES THE LIMITATIONS**

↓

**CURRENT IMPLEMENTATION**

↓

**EXPERIMENTAL RESULTS OR HONEST CURRENT STATUS**

↓

**REMAINING WORK**

The panel should be able to understand the entire research progression by looking at the figures even before reading every bullet.

Prioritize **research-specific figures over decorative visuals**.

Do not fabricate any figure, result, metric, architecture component, or experimental outcome.