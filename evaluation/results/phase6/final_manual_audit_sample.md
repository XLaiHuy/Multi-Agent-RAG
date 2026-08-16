# Phase 6 Final Held-Out Manual Audit Sample ($N=20$)

This document contains a stratified sample of 20 real API evaluations (10 Answerable, 10 Unanswerable) from the Phase 6 Final Benchmark run on 25 unseen contracts.

---

### Sample #01: `test_v2_cuad_cuad_contract_056_WA_Agreement_Date_2`
- **Question**: What is the effective execution date of this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `19  DAY OF MAY , 2010`
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The Agreement shall commence on the Commencement Date [Reference 3: cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03_v1_p1_c3], where "Commencement Date" is defined as the date of this Agreement [Reference 2: cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03_v1_p0_c2]. Additionally, the Agreement shall not take effect until it has been executed by both Parties [Reference 1: cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03_v1_p8_c1].
- **Citations Included**: 3
  - `[cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03_v1_p0_c2]` (Doc: `cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03`)
  - `[cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03_v1_p1_c3]` (Doc: `cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03`)
  - `[cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03_v1_p8_c1]` (Doc: `cuad_contract_056_WASTE2ENERGYHOLDINGS_INC_06_03`)
- **Telemetry**: 5301 tokens | 64994.5 ms

---

### Sample #02: `test_v2_cuad_cuad_contract_053_Ve_Document_Name_0`
- **Question**: What is the official title and full document name of this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `TRANSFER AND SERVICING AGREEMENT, d`
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3698 tokens | 31196.9 ms

---

### Sample #03: `test_v2_cuad_cuad_contract_042_Im_Document_Name_0`
- **Question**: What is the official title and full document name of this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `CO-BRANDING AGREEMENT`
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The official title and full document name of this agreement is "Exhibit 10.11 CO-BRANDING AGREEMENT" [Reference 5: cuad_contract_042_ImpresseCorp_20000322_S_1A_EX__v1_p0_c0].
- **Citations Included**: 1
  - `[cuad_contract_042_ImpresseCorp_20000322_S_1A_EX__v1_p0_c0]` (Doc: `cuad_contract_042_ImpresseCorp_20000322_S_1A_EX_`)
- **Telemetry**: 5486 tokens | 35805.0 ms

---

### Sample #04: `test_v2_cuad_cuad_contract_054_VE_Agreement_Date_2`
- **Question**: What is the effective execution date of this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `May 6, 2019`
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question. While Section 1.16 states that "Effective Date" has the meaning set forth in the Preamble [Reference 4: cuad_contract_054_VERICELCORP_08_06_2019_EX_10_1_v1_p1_c1], the Preamble itself is not included in the provided text.
- **Citations Included**: 1
  - `[cuad_contract_054_VERICELCORP_08_06_2019_EX_10_1_v1_p1_c1]` (Doc: `cuad_contract_054_VERICELCORP_08_06_2019_EX_10_1`)
- **Telemetry**: 3152 tokens | 25415.2 ms

---

### Sample #05: `test_v2_cuad_cuad_contract_059_Az_Parties_1`
- **Question**: Who are the named parties entering into this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `ATR`
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The named parties entering into this agreement are:

*   **AZUL LINHAS AÉREAS BRASILEIRAS S/A** (also referred to as "the Company" or "AZUL") [Reference 4: cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9_v1_p0_c1; Reference 5: cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9_v1_p2_c4]
*   **AVIONS DE TRANSPORT REGIONAL, G.I.E.** (also referred to as "the Repairer" or "ATR"), a French groupement d'intérêt économique [Reference 4: cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9_v1_p0_c1; Reference 5: cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9_v1_p2_c4]
- **Citations Included**: 2
  - `[cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9_v1_p0_c1]` (Doc: `cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9`)
  - `[cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9_v1_p2_c4]` (Doc: `cuad_contract_059_AzulSa_20170303_F_1A_EX_10_3_9`)
- **Telemetry**: 5729 tokens | 38512.0 ms

---

### Sample #06: `test_v2_cuad_cuad_contract_042_Im_Effective_Date_3`
- **Question**: What is the designated effective start date specified in the agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `March 3, 2000`
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The effective date of the agreement is March 3, 2000 [Reference 3: cuad_contract_042_ImpresseCorp_20000322_S_1A_EX__v1_p0_c0].
- **Citations Included**: 1
  - `[cuad_contract_042_ImpresseCorp_20000322_S_1A_EX__v1_p0_c0]` (Doc: `cuad_contract_042_ImpresseCorp_20000322_S_1A_EX_`)
- **Telemetry**: 5059 tokens | 31720.8 ms

---

### Sample #07: `test_v2_cuad_cuad_contract_060_Pe_Parties_1`
- **Question**: Who are the named parties entering into this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `PELICAN DELIVERS INC.`
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The named parties entering into this agreement are SEATTLE SOFTWARE DEVELOPERS, Inc. (the Developer) and PELICAN DELIVERS INC. (the Client) [Reference 1: cuad_contract_060_PelicanDeliversInc_20200211_S__v1_p4_c0].
- **Citations Included**: 1
  - `[cuad_contract_060_PelicanDeliversInc_20200211_S__v1_p4_c0]` (Doc: `cuad_contract_060_PelicanDeliversInc_20200211_S_`)
- **Telemetry**: 4785 tokens | 31161.8 ms

---

### Sample #08: `test_v2_cuad_cuad_contract_045_Cr_Parties_1`
- **Question**: Who are the named parties entering into this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `you as an "Affiliate"`
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The named parties entering into this agreement are Chase Bank USA, N.A. ("Chase") and the individual or entity acting as an "Affiliate" in the Chase Affiliate Program [Reference 3: cuad_contract_045_CreditcardscomInc_20070810_S_1_v1_p0_c0].
- **Citations Included**: 1
  - `[cuad_contract_045_CreditcardscomInc_20070810_S_1_v1_p0_c0]` (Doc: `cuad_contract_045_CreditcardscomInc_20070810_S_1`)
- **Telemetry**: 5264 tokens | 33733.8 ms

---

### Sample #09: `test_v2_cuad_cuad_contract_043_Al_Document_Name_0`
- **Question**: What is the official title and full document name of this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `JOINT CONTENT LICENSE AGREEMENT`
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The official title and full document name of this agreement is the JOINT CONTENT LICENSE AGREEMENT [Reference 4: cuad_contract_043_AlliedEsportsEntertainmentInc__v1_p0_c0].
- **Citations Included**: 1
  - `[cuad_contract_043_AlliedEsportsEntertainmentInc__v1_p0_c0]` (Doc: `cuad_contract_043_AlliedEsportsEntertainmentInc_`)
- **Telemetry**: 5189 tokens | 41618.6 ms

---

### Sample #10: `test_v2_cuad_cuad_contract_055_WP_Document_Name_0`
- **Question**: What is the official title and full document name of this agreement?
- **Ground Truth Answerable**: True
- **Gold Evidence**: `SERVICE AGREEMENT`
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3244 tokens | 31804.9 ms

---

### Sample #11: `test_v2_cuad_cuad_contract_052_Le_unans_Right_Of_First_Refusal_16`
- **Question**: Is there a right of first refusal, first offer, or first negotiation granted?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3340 tokens | 26787.1 ms

---

### Sample #12: `test_v2_cuad_cuad_contract_041_NI_unans_General_8`
- **Question**: Highlight the parts (if any) of this contract related to "Most Favored Nation" that should be reviewed by a lawyer. Details: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3721 tokens | 30206.0 ms

---

### Sample #13: `test_v2_cuad_cuad_contract_058_FE_unans_Parties_10`
- **Question**: Who are the named parties entering into this agreement?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `ANSWER`
- **Generated Answer**:
> The named parties entering into this Second Amendment are FEDERATED INVESTMENT MANAGEMENT COMPANY, a Delaware statutory trust (the "Adviser"), and FEDERATED ADVISORY SERVICES COMPANY, a Delaware statutory trust ("FASC") [Reference 1: cuad_contract_058_FEDERATEDGOVERNMENTINCOMESECUR_v1_p0_c0].
- **Citations Included**: 1
  - `[cuad_contract_058_FEDERATEDGOVERNMENTINCOMESECUR_v1_p0_c0]` (Doc: `cuad_contract_058_FEDERATEDGOVERNMENTINCOMESECUR`)
- **Telemetry**: 3241 tokens | 38760.4 ms

---

### Sample #14: `test_v2_cuad_cuad_contract_051_Pr_unans_Expiration_Date_4`
- **Question**: When does the initial term of this agreement expire or terminate?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 2641 tokens | 24651.0 ms

---

### Sample #15: `test_v2_cuad_cuad_contract_061_OR_unans_Notice_Period_To_Terminate_Renewal_6`
- **Question**: What is the notice period required to cancel or prevent automatic renewal?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> The provided contract excerpts do not contain information regarding a notice period required to prevent automatic renewal. While the Agreement states that the term is for 12 months and "may be renewed upon mutual, written agreement of the parties," it does not specify a mechanism or notice period for preventing an automatic renewal [Reference 1: cuad_contract_061_ORBSATCORP_08_17_2007_EX_7_3_S_v1_p0_c3]. The excerpts do mention that "Either party may terminate this Agreement at any time with 30 days written notice" [Reference 1: cuad_contract_061_ORBSATCORP_08_17_2007_EX_7_3_S_v1_p0_c3].
- **Citations Included**: 1
  - `[cuad_contract_061_ORBSATCORP_08_17_2007_EX_7_3_S_v1_p0_c3]` (Doc: `cuad_contract_061_ORBSATCORP_08_17_2007_EX_7_3_S`)
- **Telemetry**: 3410 tokens | 28937.3 ms

---

### Sample #16: `test_v2_cuad_cuad_contract_051_Pr_unans_Notice_Period_To_Terminate_Renewal_6`
- **Question**: What is the notice period required to cancel or prevent automatic renewal?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3178 tokens | 29662.2 ms

---

### Sample #17: `test_v2_cuad_cuad_contract_060_Pe_unans_Notice_Period_To_Terminate_Renewal_6`
- **Question**: What is the notice period required to cancel or prevent automatic renewal?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3011 tokens | 25162.2 ms

---

### Sample #18: `test_v2_cuad_cuad_contract_043_Al_unans_General_8`
- **Question**: Highlight the parts (if any) of this contract related to "Most Favored Nation" that should be reviewed by a lawyer. Details: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3805 tokens | 35668.2 ms

---

### Sample #19: `test_v2_cuad_cuad_contract_048_CU_unans_Parties_10`
- **Question**: Who are the named parties entering into this agreement?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `ANSWER`
- **Generated Answer**:
> Based on the provided contract excerpts, the named parties entering into the agreement are:

*   **The Servicer:** Defined in the preamble of the Agreement [Reference 5: cuad_contract_048_CUROGROUPHOLDINGSCORP_05_04_20_v1_p6_c2].
*   **The Owner:** Defined in the preamble of the Agreement [Reference 5: cuad_contract_048_CUROGROUPHOLDINGSCORP_05_04_20_v1_p6_c2].
*   **CURO RECEIVABLES FINANCE II, LLC:** A Delaware limited liability company, which is a party to the agreement [Reference 3: cuad_contract_048_CUROGROUPHOLDINGSCORP_05_04_20_v1_p6_c0].
- **Citations Included**: 2
  - `[cuad_contract_048_CUROGROUPHOLDINGSCORP_05_04_20_v1_p6_c0]` (Doc: `cuad_contract_048_CUROGROUPHOLDINGSCORP_05_04_20`)
  - `[cuad_contract_048_CUROGROUPHOLDINGSCORP_05_04_20_v1_p6_c2]` (Doc: `cuad_contract_048_CUROGROUPHOLDINGSCORP_05_04_20`)
- **Telemetry**: 6052 tokens | 53336.1 ms

---

### Sample #20: `test_v2_cuad_cuad_contract_057_WO_unans_General_14`
- **Question**: Highlight the parts (if any) of this contract related to "Non-Disparagement" that should be reviewed by a lawyer. Details: Is there a requirement on a party not to disparage the counterparty?
- **Ground Truth Answerable**: False
- **Gold Evidence**: ``
- **System Decision**: `INSUFFICIENT_EVIDENCE`
- **Generated Answer**:
> INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question.
- **Citations Included**: 0
- **Telemetry**: 3706 tokens | 33656.3 ms

---
