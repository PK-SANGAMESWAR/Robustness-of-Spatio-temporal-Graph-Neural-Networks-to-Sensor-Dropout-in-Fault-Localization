# THESIS WRITING RULES

## 1. Task

1. Write the M.Tech thesis for the fault localization project.
2. Use the template `MTech_thesis.tex` with the `ugthesis` class.
3. Make the main text 45 to 60 pages.
4. Write about all the work that is complete until now.

## 2. Read all files first

1. Read all project files before you write a chapter.
2. Read each file fully. Do not read only a part of a file.
3. Read these files:
   - The published paper (Fault-Main.pdf).
   - The specification and all stage files (00 to 14).
   - The metrics file.
   - The result files (JSON, CSV, summary docs).
   - The code, when you must know how a method works.
4. Use only the measured values from these files.
5. Do not invent a value, a result, a reference or a figure.
6. If two files give different values for one item, use the corrected value. Then tell me about the difference.
7. Do not use a claim that a file marks as withdrawn or retracted.
   - Example: Do not write that PSM is better than STGNN at k = 0. A file withdrew this claim.

## 3. Work that is not complete

1. Write only about work that is complete and has measured results.
2. Do not write sections, tables or figures for work that is in progress.
   - Example: PC-STGNN training and evaluation are in progress. Do not write their results.
3. Do not leave empty space, empty tables, "TBD", "to be added" or placeholder text.
4. You can write about work in progress only in the "Future Work" section, in a few sentences. Do not give values for it.

## 4. Content that you must not write in the thesis

1. Do not write file names, folder names, file paths or script names.
   - Example: Do not write `results/...`, `scripts/...`, `.npy`, `.json`, `.py`.
2. Do not write about CLAUDE.md, the plan files or the repository.
3. Do not write internal project words. Examples:
   - "Gate 0", "Gate 2", "Stage C", "S2.6", "R0ⁿ", "B2ᵃ".
   - "[Corrected 2026-09-23]" and other correction tags.
   - "GO / NO-GO", "pre-registration commit", "author review".
4. Change internal names to normal research words.
   - Example: Change "S2.6 inference-only fusion" to "inference-time physics fusion".
5. Do not write the dates of the internal work.

## 5. Write one part at a time

1. Write one chapter at a time. Use this sequence:
   1. Introduction
   2. Literature Survey
   3. Proposed Work (data, models, methods)
   4. Results and Inferences
   5. Conclusion and Future Enhancements
   6. Abstract, abbreviations, list of symbols, title page fields
2. Write the full chapter. Do not write a short draft.
3. After you write the chapter, compile the full document.
4. Check the chapter (see Section 6 and Section 7).
5. Tell me:
   - What the chapter contains.
   - The number of pages of the chapter.
   - The compile result and the warnings.
   - The problems that you found and fixed.
6. Stop. Do not start the next chapter until I approve the current chapter.
7. Add figure where ever needed, add a placeholder there like this is the figure mentioned
8. There is still training taking place so for the results and the values not present lets leave spaces with placeholders.

## 6. Spelling and language

1. Make sure that there are no spelling errors.
2. Do a spell check on each chapter with a spell-check tool (for example, aspell or hunspell). Ignore LaTeX commands.
3. Read the spell-check output. Correct each real error.
4. Use one spelling variant in all of the thesis. Use the American variant, the same as the paper (for example, "normalized", not "normalised").
5. Write the same name for each item in all chapters (for example, "RGSAGE-Max", "µPMU", "macro-F1").
6. Write in formal, academic English. Do not copy full paragraphs from the paper. Write the text again.

## 7. Alignment and format

1. Make sure that the alignment is correct.
2. Compile with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
3. Get zero errors.
4. Get zero "Overfull \hbox" warnings. Correct each one.
5. Get zero undefined references and zero undefined citations.
6. Make sure that each table fits in the text width. Align the columns correctly. Align numbers on the decimal point where possible.
7. Make sure that each figure is centered, has a caption and a label, and is referred to in the text.
8. Put the table caption above the table. Put the figure caption below the figure.
9. Number each equation. Align multi-line equations with `align`.
10. Do not use manual spacing (`\\`, `\vspace`) to correct alignment. Use the correct environment.
11. Make sure that each chapter starts on a new page.
12. Convert the pages of the chapter to images. Look at each page. Find alignment problems and correct them.

## 8. Template rules

1. Remove all sample text from the template (beam, PSO, structural identification, Lorem Ipsum).
2. Remove the sample abbreviations and symbols. Add only the abbreviations and symbols that the thesis uses.
3. Use `\ac{}` for each abbreviation.
4. Put all references in `thesis_refer.bib`. Remove the old sample entries.
5. Use `\cite{}`, `\citet{}` or `\citep{}` for each reference.
6. Use only references that are in the paper or in the project files. Do not invent a reference.
7. Keep the last paragraph of the Introduction as a summary of the remaining chapters.