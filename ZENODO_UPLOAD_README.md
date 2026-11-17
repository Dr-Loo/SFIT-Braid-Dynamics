# Zenodo deposit draft — Stage 5G working note v2

**Publication recommendation:** Working paper; create a Zenodo draft now, and publish when underlying v2 primary files and hashes have been verified. No publication or DOI has been created.

## Metadata to paste into Zenodo

**Resource type:** Working paper (Publication)  
**Title:** SFIT Part V / Stage 5G: Theorem A: Completed-Block Stability and the Adjacent-Cluster Problem  
**Creator:** Luiz Puodzius  
**Version:** v2, manuscript revision  
**Language:** English  
**Publication date:** Actual first public release date (do not backdate to the manuscript revision).  
**License:** CC BY 4.0 proposed for manuscript, subject to author's choice. License code separately if included.

### Description / abstract

This working note studies the three-defect isotropic heading-refresh substrate of SFIT Part V. It proves an averaged completed-block stability estimate for an isolated interrupted relaxation: the probability that the total signed over/under crossing reward differs from the ballistic reference is bounded by Cq[1 + log(1/q)]. The proof combines longitudinal anti-concentration for independent uniform headings, the chord lemma for repeated-defect renewals, and an oriented-area sign analysis for overlapping motion of different defects. The estimate establishes absolute convergence of isolated single-insertion response integrals and an O(r^2) Beta-law correction; it does not establish differentiability of the full stationary Green–Kubo coefficient. The note formulates the adjacent two-gap mixed response, identifies the residual displacement proportional to uv, and states the unproved adjacent-stability, higher-cluster, and stationary-limit estimates needed for a full small-r expansion theorem. A separate event-driven v2 primary test of the candidate second-order crossing-count coefficient, nu_2 = 3.222716 (numerical integration SE 0.025403), agrees with a fresh-seed extrapolated estimate 3.2301 (statistical SE 0.0404), with a prespecified decision statistic z = 0.15 (PASS). This is coefficient-level numerical validation under the stated decision rule, not a proof of the full stationary expansion. The second-order Green–Kubo coefficient Gamma_2 remains a numerical candidate.

### Keywords

SFIT, stochastic braid substrate, Green–Kubo persistence, ballistic limit, heading-refresh dynamics, crossing statistics, completed-block stability, Markov operator, renewal processes, event-driven Monte Carlo, cluster expansion

### Important versioning distinction

If Stage 5G v1 already has a *published Zenodo record*, use its **New version** feature to preserve Zenodo's linked version history. If v1 exists only as a local file or draft, this is a **new record** whose title/version may still say manuscript v2. Add only actual related-record DOI identifiers, not inferred ones.

## Published claims / caveats

- **Proved, isolated model:** Theorem A, P(Delta_u L_k != 0 | i,j) <= C u(1+ln(1/u)).
- **Established for isolated insertion:** Absolute convergence and O(r^2) Beta-law correction.
- **v2 numerically validated:** Candidate nu2=3.222716 +- 0.025403 (numerical integration uncertainty); primary extrapolation 3.2301 +- 0.0404 (stat) +- 0.0042 (nu1); predeclared score z=0.15, PASS, as reported by the v2 run.
- **Numerical candidate only:** Gamma2 approximately 0.9426 +- 0.0956; variance-specific components untested.
- **Unproved:** Uniform adjacent stability (B), higher connected-cluster growth, stationary-limit interchange, and the global O(r^2) Gamma expansion theorem.

## File and provenance readiness

Files currently verified in this local bundle:

| File | SHA-256 |
|---|---|
| `stage5g_derivation_note_v2.pdf` | `9ace835ead8cad84279cffb07fb704c63bbb69fa204341a3bd927482ef0409ab` |
| `stage5g_derivation_note_v2.tex` | `52000c20e280905746205b0d1761dffb7ad993330f1d3fc03dbb90cac4b4a1fd` |
| `zenodo_metadata_draft.json` | `4b71d6471d7c1d669ed79ff5385c791cda582051522dd66b527bf4386c895e0b` |

The following v2 hashes are transcribed from the user's primary run and **not independently verified against their files**:

- Simulator: `547126a9667714becb03ebde9799ef0b99b92cb0137f6b34ead08d10ee6621ed`
- Predictions: `4a6e9d9a28e6fdbb10039b49693c1401ad824a2c7fee92dbbac60c98ee554229`

To lock numerical provenance, add the actual simulator and frozen prediction JSON plus v2 primary result JSON, groups CSV, seed/run log, decision-rule record, and their computed checksums **before publishing**. The manuscript PDF alone documents the reported hashes but does not make these numerical results reproducible.

## Publication checklist

1. Verify listed manuscript file checksums, add primary v2 source/result files, and audit their reported hashes.
2. Confirm author display name, optional ORCID/affiliation, whether CC BY 4.0 is appropriate, and parent Part V DOI if any.
3. Create a Zenodo **draft** record and paste these metadata; upload verified files (PDF, source, v2 provenance package).
4. Preview the abstract, rights, and files; optionally reserve a DOI before finalization.
5. Publish only after confirming the precise separation between proof, prospective numerical validation, and remaining conjectures.
