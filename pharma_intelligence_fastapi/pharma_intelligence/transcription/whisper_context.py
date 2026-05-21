PHARMA_WHISPER_PROMPT = (
    "This is a pharmaceutical presentation or recording discussing "
    "drug names, active pharmaceutical ingredients, dosages, clinical trials, "
    "regulatory submissions, pharmacokinetics, pharmacodynamics, adverse events, "
    "contraindications, drug interactions, bioavailability, metabolism, "
    "medical terms, patient outcomes, and treatment protocols. "
    "Common drug names include metformin, atorvastatin, lisinopril, amlodipine, "
    "omeprazole, simvastatin, losartan, albuterol, gabapentin, sertraline, "
    "fluticasone, rosuvastatin, pantoprazole, levothyroxine, amoxicillin, "
    "azithromycin, ciprofloxacin, prednisone, warfarin, clopidogrel, insulin. "
    "Dosage formats include mg, mcg, ml, mg/kg, mg/day, units, IU, mmol. "
    "Drug forms include tablets, capsules, injections, suspensions, patches, "
    "inhalers, creams, ointments, solutions, and suppositories."
)


def get_pharma_prompt() -> str:
    return PHARMA_WHISPER_PROMPT
